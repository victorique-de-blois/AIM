import io
import logging
import os
import pathlib
import time
import warnings
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import gym
import gymnasium
from torch.nn import functional as F
from aim.sb3.common.noise import ActionNoise, VectorizedActionNoise
from aim.sb3.common.base_class import BaseAlgorithm
from aim.sb3.common.callbacks import BaseCallback
from aim.sb3.common.buffers import DictReplayBuffer, ReplayBuffer
from aim.sb3.common.save_util import load_from_pkl, save_to_pkl
from aim.sb3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from aim.sb3.common.utils import polyak_update
from aim.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from aim.sb3.td3.td3 import TD3
from aim.bc_trainer import BCTrainer
import multiprocessing as mp
from aim.sb3.common.utils import safe_mean, should_collect_more_steps
from aim.sb3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, idx: int, model: BCTrainer) -> None:
        parent_remote.close()
        while True:
            cmd, data = remote.recv()
            if cmd == "predict_actor":
                remote.send(model.policy.predict(*data))
            elif cmd == "predict_critic":
                remote.send(model.critic(*data)[0].detach())
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "set_training_mode":
                remote.send(model.policy.set_training_mode(data))
            elif cmd == "train":
                remote.send(model.train(*data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")

class AIM(TD3):
    def __init__(self, num_instances=1, q_value_bound=1., *args, **kwargs):
        self.k = num_instances
        self.classifier = kwargs.get("classifier")
        self.init_bc_steps=kwargs.pop("init_bc_steps")
        self.ensembles = [BCTrainer(seed=_, *args, **kwargs) for _ in range(num_instances)]
        self.actors = th.nn.ModuleList([self.ensembles[_].actor for _ in range(num_instances)])
        self.critics = th.nn.ModuleList([self.ensembles[_].critic for _ in range(num_instances)])
        self.num_gd = 0
        self.next_update = 0
        self.estimates = []

        self.extra_config = {}
        for k in ["thr_classifier"]:
            if k in kwargs:
                self.extra_config[k] = kwargs.pop(k)

        self.q_value_bound = q_value_bound
        
        super(AIM, self).__init__(seed=0, *args, **kwargs)
        
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["actors", "critics"]
        return state_dicts, []
    
    def _excluded_save_params(self) -> List[str]:
        return super(AIM, self)._excluded_save_params() + [
            "ensembles", "remotes", "work_remotes", "processes"
        ]
    
    def critic_all(self,
        observation: th.Tensor, action: th.Tensor) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        assert observation.ndim == action.ndim
        values = []
        for remote in self.remotes:
            remote.send(("predict_critic", (observation, action)))
        for remote in self.remotes:
            values.append(remote.recv())
        return th.stack(values).to(self.device)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        policy_choice: Optional[np.ndarray] = None,
        return_all: bool = False
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if policy_choice == None:
            actions, states_ = [], state
            for remote in self.remotes:
                remote.send(("predict_actor", (observation, state, episode_start, deterministic)))
            for remote in self.remotes:
                action, _ = remote.recv()
                actions.append(action)
            if return_all:
                npactions = np.array(actions)
            else:
                npactions = np.array(actions).mean(axis=0)
            return npactions, states_
        
        actions, states_ = [], state
        for idx, choice in enumerate(policy_choice):
            remote = self.remotes[choice]
            remote.send(("predict_actor", (observation[idx], state, episode_start, deterministic)))
            action, _ = remote.recv()
            actions.append(action)
        npactions = np.array(actions)
        return npactions, states_
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
        deterministic=None,
        policy_choice: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            if deterministic is None:
                deterministic = False
            unscaled_action, _ = self.predict(self._last_obs, deterministic=deterministic, policy_choice=policy_choice)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, (gym.spaces.Box, gymnasium.spaces.Box)):
            low, high = self.action_space.low, self.action_space.high
            scaled_action = 2.0 * ((unscaled_action - low) / (high - low)) - 1.0

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action

            action = low + (0.5 * (scaled_action + 1.0) * (high - low))
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def compute_unc(self, obs: np.ndarray):
        all_actions, _ = self.predict(observation=obs, return_all=True)
        estim = all_actions.var(axis = 0).mean()
        return estim
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        deterministic=None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        
        #for remote in self.remotes:
        #    remote.send(("set_training_mode", False))
        #for remote in self.remotes:
        #    remote.recv()

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            raise NotImplementedError
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        self.policy_choice = np.random.randint(self.k, size=env.num_envs)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                raise NotImplementedError
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(
                learning_starts, action_noise, env.num_envs, deterministic=deterministic, policy_choice=None #self.policy_choice
            )

            if hasattr(self, "trained"):
                th_obs = th.from_numpy(self._last_obs).to(self.device)
                th_actions = th.from_numpy(actions).to(self.device)
                unc = self.classifier.critic(th_obs, th_actions)[0].item()

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            
            if hasattr(self, "trained") and not infos[0]["takeover"]:
                self.estimates.append(unc)

            self.num_timesteps += env.num_envs
            self.since_last_reset += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(
                    num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False
                )

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            self._on_step()

            for idx, done in enumerate(dones):
                if done:

                    self.policy_choice[idx] = np.random.randint(self.k)

                    if len(self.estimates) > 25:
                        self.switch2human_thresh = th.quantile(th.Tensor(self.estimates).squeeze(), self.extra_config["thr_classifier"]).item()
                        self.uthred = [self.switch2human_thresh, self.switch2robot_thresh]

                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)
                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()

                    # PZH: We add a callback here to allow doing something after each episode is ended.
                    self._on_episode_end(infos[idx])

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        stat_recorder = defaultdict(float)
        stat_recorders = []
        lm = batch_size // self.env.num_envs
        
        if not hasattr(self, "trained"):
            stat_recorder["wall_steps"] = self.num_timesteps
        else:
            stat_recorder["wall_steps"] = self.human_data_buffer.pos - self.initial_demos + self.init_bc_steps
        
        if self.num_timesteps >= self.init_bc_steps - 1 and not hasattr(self, "trained"):
            self.trained = True
            #thompson sample
            lm = self.human_data_buffer.pos
            
            self.initial_demos = lm
            
            len_train = (int)(0.9 * lm)
            tmp_buffers_id = []
            for remote in self.remotes:
                tmp_buffers_id.append(np.random.randint(0, len_train, size=len_train))

            ##first training of student policies: bc
            for _ in range(self.init_bc_steps):
                stat_recorders = []
                self.num_gd += 1
                for t_, remote in enumerate(self.remotes):
                    idx = np.random.randint(0, len_train, size=batch_size)
                    replay_data = self.human_data_buffer._get_samples(tmp_buffers_id[t_][idx], env=self._vec_normalize_env)
                    remote.send(("train", (gradient_steps, batch_size, replay_data)))
                for remote in self.remotes:
                    stat_recorders.append(remote.recv())
            ##start train classifier
            num_gd_steps = self.init_bc_steps
            for _ in range(num_gd_steps):
                    with th.no_grad():
                        replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                        new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                            
                    current_c_behavior = self.classifier.critic(replay_data_human.observations, replay_data_human.actions_behavior)[0]
                    current_c_novice = self.classifier.critic(replay_data_human.observations, th.Tensor(new_action).to(self.device))[0]
                    loss_class = th.mean((current_c_behavior + 1) ** 2 + (current_c_novice - 1) ** 2)
                        
                    self.classifier.critic.optimizer.zero_grad()
                    loss_class.backward()
                    self.classifier.critic.optimizer.step()
            stat_recorder["loss_class"] = loss_class.item()
            self.estimates = []

            train_data = self.human_data_buffer._get_samples(np.arange(len_train), env=self._vec_normalize_env)
            val_data = self.human_data_buffer._get_samples(len_train + np.arange(lm - len_train), env=self._vec_normalize_env)

            all_actions, _ = self.predict(observation=val_data.observations.cpu().numpy(), return_all=True)
            heldout_estim = th.Tensor(all_actions.var(axis = 0).mean(axis = -1)).to(self.device)

            actions_train_data, _ = self.predict(observation=train_data.observations.cpu().numpy())
            actions_train_data = th.Tensor(actions_train_data).to(self.device)
            discre = th.mean((train_data.actions_behavior - actions_train_data) ** 2, dim = -1)

            self.switch2robot_thresh = th.mean(discre).item()
            with th.no_grad():
                replay_data_human = val_data
                new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                current_c_novice = self.classifier.critic(replay_data_human.observations, th.Tensor(new_action).to(self.device))[0]
            self.estimates = current_c_novice.squeeze().tolist()
            self.switch2human_thresh = th.quantile(current_c_novice, self.extra_config["thr_classifier"]).item()
            self.next_update = self.human_data_buffer.pos + self.policy_delay
            
        
        if hasattr(self, "trained") and (self.human_data_buffer.pos >= self.next_update):
            self.next_update = self.human_data_buffer.pos + self.policy_delay
            self.estimates = []
            if self.human_data_buffer.full:
                len_train = self.human_data_buffer.buffer_size
            else:
                len_train = self.human_data_buffer.pos
            tmp_buffers_id = []
            for remote in self.remotes:
                tmp_buffers_id.append(np.random.randint(0, len_train, size=len_train))
            
            for _ in range(self.policy_delay):
                stat_recorders = []
                self.num_gd += 1
                for t_, remote in enumerate(self.remotes):
                    idx = np.random.randint(0, len_train, size=batch_size)
                    replay_data = self.human_data_buffer._get_samples(tmp_buffers_id[t_][idx], env=self._vec_normalize_env)
                    remote.send(("train", (gradient_steps, batch_size, replay_data)))
                for remote in self.remotes:
                    stat_recorders.append(remote.recv())
        
        
        if hasattr(self, "trained") and (self.human_data_buffer.pos % self.policy_delay == 0):
            ##start train classifier
            for _ in range(self.policy_delay):
                    with th.no_grad():
                        replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                        new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                        new_action = th.Tensor(new_action).to(self.device)
                    current_c_behavior = self.classifier.critic(replay_data_human.observations, replay_data_human.actions_behavior)[0]
                    current_c_novice = self.classifier.critic(replay_data_human.observations, new_action)[0]
                    no_overlap = (
                        ((replay_data_human.actions_behavior - new_action) ** 2).mean(dim=-1) > self.switch2robot_thresh
                    ).float()
                    loss_class = th.mean((current_c_behavior + 1) ** 2 + (current_c_novice * no_overlap - 1) ** 2)
                    self.classifier.critic.optimizer.zero_grad()
                    loss_class.backward()
                    self.classifier.critic.optimizer.step()
            stat_recorder["loss_class"] = loss_class.item()

        for remote in self.remotes:
            remote.send(("set_training_mode", False))
        for remote in self.remotes:
            remote.recv()
        
        avg_stat = defaultdict(float)
        for stat_recorderins in stat_recorders:
            for key, values in stat_recorderins.items():
                avg_stat[key] += np.mean(values) / self.k
        
        avg_stat["human_buffer_size"] = self.human_data_buffer.pos
        avg_stat["ep_takeover"] = self.human_data_buffer.pos / self.num_timesteps
        for key, values in avg_stat.items():
            self.logger.record("train/{}".format(key), values)
        
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        
        stat_recorder["num_gd"] = self.num_gd
        if hasattr(self, "trained"):
            stat_recorder["switch2human_thresh"] = self.switch2human_thresh
            stat_recorder["switch2robot_thresh"] = self.switch2robot_thresh
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), values)
        try:
            import wandb
            wandb.log(self.logger.name_to_value, step=self.num_timesteps)
        except:
            pass
    
    def _setup_model(self) -> None:
        super(AIM, self)._setup_model()
        self.human_data_buffer = HACOReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    n_envs=self.n_envs,
                    optimize_memory_usage=self.optimize_memory_usage,
                    **self.replay_buffer_kwargs
                )
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.k)])
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            args = (work_remote, remote, i, self.ensembles[i])
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super(AIM, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)

    def save_replay_buffer(
        self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
                                                                                          io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super(AIM, self).save_replay_buffer(path_replay)

    def load_replay_buffer(
        self,
        path_human: Union[str, pathlib.Path, io.BufferedIOBase],
        path_replay: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.human_data_buffer = load_from_pkl(path_human, self.verbose)
        assert isinstance(
            self.human_data_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.human_data_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.human_data_buffer.handle_timeout_termination = False
            self.human_data_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)
        super(AIM, self).load_replay_buffer(path_replay, truncate_last_traj)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_timesteps: int = 2000,
        buffer_save_timesteps: int = 200,
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = False,
        load_buffer: bool = False,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        warmup: bool = False,
        warmup_steps: int = 5000,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
        callback.on_training_start(locals(), globals())
        if warmup:
            assert load_buffer, "warmup is useful only when load buffer"
            print("Start warmup with steps: " + str(warmup_steps))
            self.train(batch_size=self.batch_size, gradient_steps=warmup_steps)

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break
            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            if save_buffer and self.human_data_buffer.pos > 0 and self.human_data_buffer.pos % buffer_save_timesteps == 0:
                buffer_location_human = os.path.join(
                    save_path_human, "human_buffer_" + str(self.human_data_buffer.pos) + ".pkl"
                )
                buffer_location_replay = os.path.join(
                    save_path_replay, "replay_buffer_" + str(self.human_data_buffer.pos) + ".pkl"
                )
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self