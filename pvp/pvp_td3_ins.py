import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional
from torchmetrics import PrecisionRecallCurve
from torchmetrics.classification import BinaryAveragePrecision
import numpy as np
import torch as th
from torch.nn import functional as F
from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.common.utils import polyak_update
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from pvp.sb3.td3.td3 import TD3
from pvp.sb3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
logger = logging.getLogger(__name__)


class PVPTD3(TD3):
    def __init__(self, use_balance_sample=True, q_value_bound=1., *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        if "cql_coefficient" in kwargs:
            self.cql_coefficient = kwargs["cql_coefficient"]
            kwargs.pop("cql_coefficient")
        else:
            self.cql_coefficient = 1
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer

        if "intervention_start_stop_td" in kwargs:
            self.intervention_start_stop_td = kwargs["intervention_start_stop_td"]
            kwargs.pop("intervention_start_stop_td")
        else:
            # Default to set it True. We find this can improve the performance and user experience.
            self.intervention_start_stop_td = True

        self.extra_config = {}
        for k in ["no_done_for_positive", "no_done_for_negative", "reward_0_for_positive", "reward_0_for_negative",
                  "reward_n2_for_intervention", "reward_1_for_all", "use_weighted_reward", "remove_negative",
                  "adaptive_batch_size", "add_bc_loss", "only_bc_loss"]:
            if k in kwargs:
                v = kwargs.pop(k)
                assert v in ["True", "False"]
                v = v == "True"
                self.extra_config[k] = v
        for k in ["agent_data_ratio", "bc_loss_weight", "classifier", "thr_classifier"]:
            if k in kwargs:
                self.extra_config[k] = kwargs.pop(k)

        self.q_value_bound = q_value_bound
        self.use_balance_sample = use_balance_sample
        if not hasattr(self, "pvpinstance"):
            self.pvpinstance = True
        super(PVPTD3, self).__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super(PVPTD3, self)._setup_model()
        if self.pvpinstance:
            self.human_data_buffer = None
            self.replay_buffer = None
            self.all_buffer = None
        else:
            if self.use_balance_sample:
                self.human_data_buffer = HACOReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    n_envs=self.n_envs,
                    optimize_memory_usage=self.optimize_memory_usage,
                    **self.replay_buffer_kwargs
                )
                self.all_buffer = HACOReplayBuffer(
                    5000,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    n_envs=self.n_envs,
                    optimize_memory_usage=self.optimize_memory_usage,
                    **self.replay_buffer_kwargs
                )
            else:
                self.human_data_buffer = self.replay_buffer

    def train(self, gradient_steps: int, batch_size: int, replay_data: ReplayBufferSamples = None) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        #self.predictor.set_training_mode(True)
        #self.uncpolicy.set_training_mode(True)
        #self.target.set_training_mode(False)
        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer]) #, self.predictor.optimizer, self.unc.optimizer])

        stat_recorder = defaultdict(list)
        if replay_data == None:
            return stat_recorder

        lm = batch_size // self.env.num_envs
        for step in range(gradient_steps):
            self._n_updates += 1
            
            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions_behavior.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_behavior_values = self.critic(replay_data.observations, replay_data.actions_behavior)
            current_q_novice_values = self.critic(replay_data.observations, replay_data.actions_novice)

            stat_recorder["q_value_behavior"].append(current_q_behavior_values[0].mean().item())
            stat_recorder["q_value_novice"].append(current_q_novice_values[0].mean().item())

            # Compute critic loss
            critic_loss = []
            for (current_q_behavior, current_q_novice) in zip(current_q_behavior_values, current_q_novice_values):
                if self.intervention_start_stop_td:
                    l = 0.5 * F.mse_loss(
                        replay_data.stop_td * current_q_behavior, replay_data.stop_td * target_q_values
                    )

                else:
                    l = 0.5 * F.mse_loss(current_q_behavior, target_q_values)

                # ====== The key of Proxy Value Objective =====
                l += th.mean(
                    replay_data.interventions * self.cql_coefficient *
                    (F.mse_loss(current_q_behavior, self.q_value_bound * th.ones_like(current_q_behavior)))
                )
                l += th.mean(
                    replay_data.interventions * self.cql_coefficient *
                    (F.mse_loss(current_q_novice, -self.q_value_bound * th.ones_like(current_q_novice)))
                )

                critic_loss.append(l)
            critic_loss = sum(critic_loss)

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            stat_recorder["critic_loss"].append(critic_loss.item())

            if True:
                # Compute actor loss
                a_pred = self.actor(replay_data.observations)
                loss_pi = th.mean((replay_data.actions_behavior - a_pred)**2)
                self.actor.optimizer.zero_grad()
                loss_pi.backward()
                self.actor.optimizer.step()
                stat_recorder["loss_pi"].append(loss_pi.item())

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            
            
        return stat_recorder

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
        super(PVPTD3, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos, self.all_buffer)

    def save_replay_buffer(
        self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
                                                                                          io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super(PVPTD3, self).save_replay_buffer(path_replay)

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
        super(PVPTD3, self).load_replay_buffer(path_replay, truncate_last_traj)

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
                buffer_location_all = os.path.join(
                    save_path_replay, "all_buffer_" + str(self.human_data_buffer.pos) + ".pkl"
                )
                save_to_pkl(buffer_location_all, self.all_buffer, self.verbose)
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self