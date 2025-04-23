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
from pvp.sb3.common.noise import ActionNoise, VectorizedActionNoise
from pvp.sb3.common.base_class import BaseAlgorithm
from pvp.sb3.common.callbacks import BaseCallback
from pvp.sb3.common.buffers import DictReplayBuffer, ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from pvp.sb3.td3.td3 import TD3
from pvp.pvp_td3_ins import PVPTD3
import multiprocessing as mp
from pvp.sb3.common.utils import safe_mean, should_collect_more_steps
from pvp.sb3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)

def _worker(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, idx: int, model: PVPTD3) -> None:
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

class PVPTD3ENS(PVPTD3):
    def __init__(self, num_instances=1, *args, **kwargs):
        self.k = num_instances
        self.classifier = kwargs.get("classifier")
        self.ensembles = [PVPTD3(seed=_, *args, **kwargs) for _ in range(num_instances)]
        self.actors = th.nn.ModuleList([self.ensembles[_].actor for _ in range(num_instances)])
        self.critics = th.nn.ModuleList([self.ensembles[_].critic for _ in range(num_instances)])
        self.pvpinstance = False
        self.num_gd = 0
        self.next_update = 0
        self.estimates = []
        super(PVPTD3ENS, self).__init__(seed=0, *args, **kwargs)
    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["actors", "critics"]
        return state_dicts, []
    def _excluded_save_params(self) -> List[str]:
        return super(PVPTD3ENS, self)._excluded_save_params() + [
            "ensembles", "remotes", "work_remotes", "processes"
        ]
    def _setup_model(self) -> None:
        super(PVPTD3ENS, self)._setup_model()
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

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
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
        dd = ["takeover_current", "total_switch", "total_miss", "total_colorchange", "takeover"]
        for key in dd:
            if hasattr(self, key):
                stat_recorder[key].append(getattr(self, key))
        
        self.init_bc_steps = 200
        
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
            num_gd_steps = self.init_bc_steps #self.policy_delay
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
            #change train_data.actions_novice to new actions  chy: 0108
            
            self.switch2robot_thresh = th.mean(discre).item()
            # self.switch2human_thresh = th.quantile(heldout_estim, 0.95).item()
            # self.uthred = [self.switch2human_thresh, self.switch2robot_thresh]
            # self.human_data_buffer.pos = len_train
            with th.no_grad():
                replay_data_human = val_data #self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                current_c_novice = self.classifier.critic(replay_data_human.observations, th.Tensor(new_action).to(self.device))[0]
            self.estimates = current_c_novice.squeeze().tolist()
            self.switch2human_thresh = th.quantile(current_c_novice, self.extra_config["thr_classifier"]).item()
            self.next_update = self.human_data_buffer.pos + self.policy_delay
            
        
        if hasattr(self, "trained") and (self.human_data_buffer.pos >= self.next_update):
            self.next_update = self.human_data_buffer.pos + self.policy_delay
            self.estimates = []
            ##Now we do not retrain pi from scratch for fairness
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
        
        
        if hasattr(self, "trained") and (self.human_data_buffer.pos % (8 * self.policy_delay) == 0):
            ##start train classifier
            num_gd_steps = self.policy_delay * 8
            for _ in range(num_gd_steps):
                    with th.no_grad():
                        replay_data_human = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                        new_action, _ = self.predict(replay_data_human.observations.cpu().numpy())
                        new_action = th.Tensor(new_action).to(self.device)
                            
                    current_c_behavior = self.classifier.critic(replay_data_human.observations, replay_data_human.actions_behavior)[0]
                    
                    current_c_novice = self.classifier.critic(replay_data_human.observations, new_action)[0]
                    
                    no_overlap = (
                        ((replay_data_human.actions_behavior - new_action) ** 2).mean(dim=-1) > self.switch2robot_thresh * 1.5
                    ).float()
                    
                    loss_class = th.mean((current_c_behavior + 1) ** 2 + (current_c_novice * no_overlap - 1) ** 2)
                        
                    self.classifier.critic.optimizer.zero_grad()
                    loss_class.backward()
                    self.classifier.critic.optimizer.step()
            stat_recorder["loss_class"] = loss_class.item()
            ## change to quantile

        for remote in self.remotes:
            remote.send(("set_training_mode", False))
        for remote in self.remotes:
            remote.recv()
        
        avg_stat = defaultdict(float)
        for stat_recorderins in stat_recorders:
            for key, values in stat_recorderins.items():
                avg_stat[key] += np.mean(values) / self.k
        
        avg_stat["human_buffer_size"] = self.human_data_buffer.pos
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


    def train_offline(self, gradient_steps: int, batch_size: int = 100, timestep: int = 0) -> None:
        stat_recorder = defaultdict(float)
        stat_recorders = defaultdict(list)
        lm = batch_size // self.env.num_envs
        dd = ["takeover_current", "total_switch", "total_miss", "total_colorchange", "takeover"]
        for key in dd:
            if hasattr(self, key):
                stat_recorder[key].append(getattr(self, key))
        
        self.init_bc_steps = 0
        
        self.timestep = timestep
        
        self.human_data_buffer.pos = timestep
        
        stat_recorders["wall_steps"].append(timestep)
        
        if True:
            self.actor.reset_parameters()
            self.trained = True


            ##first training of student policies: bc
            import tqdm
            for _ in tqdm.trange(20000, desc="Gradient Steps"):
                replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
                a_pred = self.actor(replay_data.observations)
                loss_pi = th.mean((replay_data.actions_behavior - a_pred)**2)
                self.actor.optimizer.zero_grad()
                loss_pi.backward()
                self.actor.optimizer.step()
                stat_recorders["loss_pi"].append(loss_pi.item())

        avg_stat = defaultdict(float)
        for stat_recorderins, values in stat_recorders.items():
                avg_stat[stat_recorderins] += np.mean(values) / self.k
        
        avg_stat["human_buffer_size"] = self.human_data_buffer.pos
        for key, values in avg_stat.items():
            self.logger.record("train/{}".format(key), values)
        
        
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        
        stat_recorder["num_gd"] = self.num_gd
        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), values)
        try:
            import wandb
            wandb.log(self.logger.name_to_value, step=(int)(timestep))
        except:
            pass
        
        

    def learn_offline(
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
        buffer_save_timesteps: int = 200,
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = False,
        load_buffer: bool = True,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "/home/caihy/pvp/pvp/human_buffer_22200.pkl",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "/home/caihy/pvp/pvp/replay_buffer_4000.pkl",
    ) -> "OffPolicyAlgorithm":
        if load_buffer:
            self.load_replay_buffer(load_path_human, load_path_replay)
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

        callback.on_training_start(locals(), globals())
        while self.num_timesteps < total_timesteps:
            self.num_timesteps += 1
            self.human_data_buffer.pos = self.num_timesteps
            self.policy.set_training_mode(False)

            num_collected_steps, num_collected_episodes = 0, 0

            callback.on_rollout_start()
            continue_training = True

            if True:
                self.num_timesteps += 1
                self.since_last_reset += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                callback.on_step()

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is dones as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

            callback.on_rollout_end()
            if True:
                gradient_steps = self.gradient_steps
                assert gradient_steps > 0
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train_offline(batch_size=self.batch_size, gradient_steps=gradient_steps)
            
        callback.on_training_end()
