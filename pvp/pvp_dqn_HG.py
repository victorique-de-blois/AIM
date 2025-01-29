import io
import pathlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch as th
from torch.nn import functional as F
import torch
from pvp.sb3.common.buffers import ReplayBuffer
from pvp.sb3.common.save_util import recursive_getattr, save_to_zip_file
from pvp.sb3.dqn.dqn import DQN, compute_entropy
from pvp.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples, HACODictReplayBufferSamples
import tqdm
from pvp.sb3.dqn.policies import CnnPolicy
from pvp.sb3.common.type_aliases import GymEnv, MaybeCallback
from pvp.sb3.common.save_util import load_from_pkl, save_to_pkl
import logging
import os
logger = logging.getLogger(__name__)

def sample_and_concat(replay_data_agent, replay_data_human, agent_data_index):
    replay_data = concat_samples(HACODictReplayBufferSamples(
        observations=replay_data_agent.observations[agent_data_index],
        actions_novice=replay_data_agent.actions_novice[agent_data_index],
        next_observations=replay_data_agent.next_observations[agent_data_index],
        dones=replay_data_agent.dones[agent_data_index],
        rewards=replay_data_agent.rewards[agent_data_index],
        actions_behavior=replay_data_agent.actions_behavior[agent_data_index],
        interventions=replay_data_agent.interventions[agent_data_index],
        stop_td=replay_data_agent.stop_td[agent_data_index],
        intervention_costs=replay_data_agent.intervention_costs[agent_data_index],
        takeover_log_prob=replay_data_agent.takeover_log_prob[agent_data_index],
        next_intervention_start=replay_data_agent.next_intervention_start[agent_data_index],
        # feature_observations=replay_data_agent.feature_observations[agent_data_index] if replay_data_agent.feature_observations is not None else None,
        # feature_next_observations=replay_data_agent.feature_next_observations[agent_data_index] if replay_data_agent.feature_next_observations is not None else None,
    ), replay_data_human)
    return replay_data


class PVPDQN(DQN):
    classifier: CnnPolicy
    def __init__(self, q_value_bound=1., *args, **kwargs):
        kwargs["replay_buffer_class"] = HACOReplayBuffer
        if "replay_buffer_class" not in kwargs:
            kwargs["replay_buffer_class"] = HACOReplayBuffer
        if "adaptive_batch_size" in kwargs:
            self.adaptive_batch_size = kwargs.pop("adaptive_batch_size")
        else:
            self.adaptive_batch_size = False


        # TODO: bc_loss_weight is not used in the code.
        if "bc_loss_weight" in kwargs:
            self.bc_loss_weight = kwargs.pop("bc_loss_weight")
        else:
            self.bc_loss_weight = 0.0
        if "add_bc_loss" in kwargs:
            self.add_bc_loss = kwargs.pop("add_bc_loss")
        else:
            self.add_bc_loss = False
        
        if "policy_delay" in kwargs:
            self.policy_delay = kwargs.pop("policy_delay")
        else:
            self.policy_delay = 1
        
        if "init_bc_steps" in kwargs:
            self.init_bc_steps = kwargs.pop("init_bc_steps")
        else:
            self.init_bc_steps = 100
            
        if "thr_classifier" in kwargs:
            self.thr_classifier = kwargs.pop("thr_classifier")
        else:
            self.thr_classifier = 0.9
            
        if "gamma_classifier" in kwargs:
            self.gamma_classifier = kwargs.pop("gamma_classifier")
        else:
            self.gamma_classifier = -1
        
        self.delay = 0

        self.gradient_steps_multiplier = kwargs.pop("gradient_steps_multiplier", 1)
        super(PVPDQN, self).__init__(*args, **kwargs)
        self.q_value_bound = q_value_bound

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # TODO: hardcoded
        discard_rgb = True
        return_features = False

        gradient_steps = gradient_steps * self.gradient_steps_multiplier

        if self.human_data_buffer.pos == 0 and self.replay_buffer.pos == 0:
            return

        self.classifier.set_training_mode(True)
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        stat_recorder = defaultdict(list)

        losses = []
        entropies = []
        stat_recorder["wall_steps"] = self.num_timesteps

        if self.adaptive_batch_size:
            replay_data_human = self.human_data_buffer.sample(
                int(batch_size), env=self._vec_normalize_env, return_all=True
            )
            human_data_size = len(replay_data_human.observations)
            human_data_size = max(1, human_data_size)
            human_data_size = int(human_data_size)

            replay_data_agent = self.replay_buffer.sample(human_data_size, env=self._vec_normalize_env, return_all=True, discard_rgb=discard_rgb)

            # TODO: I think it's too complex to extract ResNet feature for now. Maybe we can do it in future.
            # use_pretrained = self.policy_kwargs["features_extractor_kwargs"].get("pretrained", False)
            # if use_pretrained:
            #     with torch.no_grad():
            #         replay_data_human = replay_data_human._replace(
            #             feature_next_observations=self.policy.q_net_target.extract_features(
            #                 replay_data_human.next_observations),
            #             feature_observations=self.policy.q_net.extract_features(replay_data_human.observations)
            #         )
            #
            #         replay_data_agent = replay_data_agent._replace(
            #             feature_next_observations=self.policy.q_net_target.extract_features(
            #                 replay_data_agent.next_observations),
            #             feature_observations=self.policy.q_net.extract_features(replay_data_agent.observations)
            #         )

            if replay_data_human.observations.shape[0] < replay_data_agent.observations.shape[0]:
                # Reduce the number of agent actions
                replay_data = None
                should_sample_agent = True
                num_agent_samples = replay_data_agent.observations.shape[0]
                num_human_samples = replay_data_human.observations.shape[0]
            else:
                replay_data = concat_samples(replay_data_agent, replay_data_human)

                should_sample_agent = False

        if self.delay % self.policy_delay == 0:
            n_upd = self.policy_delay
        else:
            n_upd = 0

        for _ in tqdm.trange(gradient_steps * n_upd, desc="Gradient Steps"):
            if self.adaptive_batch_size:
                pass

            else:
                # Sample replay buffer
                if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
                    agent_num_samples = int(min(batch_size // 2, self.replay_buffer.pos))
                    human_num_samples = int(min(batch_size // 2, self.human_data_buffer.pos))
                    replay_data_agent = self.replay_buffer.sample(agent_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                    replay_data_human = self.human_data_buffer.sample(human_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
                elif self.human_data_buffer.pos > 0:
                    human_num_samples = int(min(batch_size, self.human_data_buffer.pos))
                    replay_data = self.human_data_buffer.sample(human_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                elif self.replay_buffer.pos > 0:
                    agent_num_samples = int(min(batch_size, self.replay_buffer.pos))
                    replay_data = self.replay_buffer.sample(agent_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                else:
                    raise ValueError("No data in replay buffer")

            if self.adaptive_batch_size and should_sample_agent:
                # Sample number of human data's data from agent's data
                ind = th.randperm(num_agent_samples)[:num_human_samples]
                replay_data = sample_and_concat(replay_data_agent, replay_data_human, ind)

            # replay_data.next_observations["backbone_features"] = replay_data.feature_next_observations
            # replay_data.observations["backbone_features"] = replay_data.feature_observations

            with th.no_grad():
                # Compute the next Q-values using the target network
                # next_q_values, _ = self.q_net_target(replay_data.next_observations)
                next_q_values= self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                assert replay_data.rewards.mean().item() == 0.0
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            # current_q_values, _ = self.q_net(replay_data.observations)
            current_q_values = self.q_net(replay_data.observations)

            entropies.append(compute_entropy(current_q_values))

            # Retrieve the q-values for the actions from the replay buffer
            current_behavior_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions_behavior.long())

            current_novice_q_value_method1 = th.gather(current_q_values, dim=1, index=replay_data.actions_novice.long())

            current_novice_q_value = current_novice_q_value_method1

            mask = replay_data.interventions
            stat_recorder["no_intervention_rate"].append(mask.float().mean().item())

            no_overlap = replay_data.actions_behavior != replay_data.actions_novice
            stat_recorder["no_overlap_rate"].append(no_overlap.float().mean().item())
            stat_recorder["masked_no_overlap_rate"].append((mask * no_overlap).float().mean().item())

            pvp_loss = \
                F.mse_loss(
                    mask * current_behavior_q_values,
                    mask * self.q_value_bound * th.ones_like(current_behavior_q_values)
                ) + \
                F.mse_loss(
                    mask * no_overlap * current_novice_q_value,
                    mask * no_overlap * (-self.q_value_bound) * th.ones_like(current_novice_q_value)
                )

            # Compute Huber loss (less sensitive to outliers)
            loss_td = F.smooth_l1_loss(current_behavior_q_values, target_q_values)

            loss = loss_td.mean() + pvp_loss.mean()

            # BC loss
            lp = torch.distributions.Categorical(logits=current_q_values).log_prob(replay_data.actions_behavior.flatten())
            masked_lp = (mask.flatten() * lp.flatten()).sum() / (mask.sum() + 1e-8)
            bc_loss = -lp.mean()
            masked_bc_loss = -masked_lp

            loss = bc_loss
            stat_recorder["bc_loss"].append(bc_loss.item())
            stat_recorder["masked_bc_loss"].append(masked_bc_loss.item())

            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

            stat_recorder["q_value_behavior"].append(current_behavior_q_values.mean().item())
            stat_recorder["q_value_novice"].append(current_novice_q_value.mean().item())
        
        if self.human_data_buffer.pos == 0 or self.num_timesteps < self.init_bc_steps:
            n_upd = 0
        
        for _ in tqdm.trange(gradient_steps * n_upd, desc="Gradient Steps"):
            human_num_samples = int(min(batch_size, self.human_data_buffer.pos))
            if self.replay_buffer.pos > 0 and self.human_data_buffer.pos > 0:
                    agent_num_samples = int(min(batch_size // 2, self.replay_buffer.pos))
                    human_num_samples = int(min(batch_size // 2, self.human_data_buffer.pos))
                    replay_data_agent = self.replay_buffer.sample(agent_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                    replay_data_human = self.human_data_buffer.sample(human_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                    replay_data = concat_samples(replay_data_agent, replay_data_human)
            elif self.human_data_buffer.pos > 0:
                    human_num_samples = int(min(batch_size, self.human_data_buffer.pos))
                    replay_data = self.human_data_buffer.sample(human_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
            elif self.replay_buffer.pos > 0:
                    agent_num_samples = int(min(batch_size, self.replay_buffer.pos))
                    replay_data = self.replay_buffer.sample(agent_num_samples, env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
            else:
                    raise ValueError("No data in replay buffer")
            
            with th.no_grad():
                # Compute the next Q-values using the target network
                # next_q_values, _ = self.q_net_target(replay_data.next_observations)
                next_q_values= self.classifier.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                assert replay_data.rewards.mean().item() == 0.0
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma_classifier * next_q_values

            # Get current Q-values estimates
            # current_q_values, _ = self.q_net(replay_data.observations)
            current_q_values = self.classifier.q_net(replay_data.observations)

            stat_recorder["c_entropy"].append(compute_entropy(current_q_values))

            # Retrieve the q-values for the actions from the replay buffer
            current_behavior_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions_behavior.long())

            with th.no_grad():
                new_actions = th.unsqueeze(self.q_net._predict(replay_data.observations).long(), -1)

                #new_actions = th.randint(low=0, high=7, size=new_actions.shape).to(self.device)
            current_novice_q_value_method1 = th.gather(current_q_values, dim=1, index=new_actions)

            current_novice_q_value = current_novice_q_value_method1

            mask = replay_data.interventions
            # stat_recorder["no_intervention_rate"].append(mask.float().mean().item())

            no_overlap = replay_data.actions_behavior != new_actions
            # stat_recorder["no_overlap_rate"].append(no_overlap.float().mean().item())
            # stat_recorder["masked_no_overlap_rate"].append((mask * no_overlap).float().mean().item())

            pvp_loss = \
                F.mse_loss(
                    mask * current_behavior_q_values,
                    mask * (-self.q_value_bound) * th.ones_like(current_behavior_q_values)
                ) + \
                F.mse_loss(
                    mask * no_overlap * current_novice_q_value,
                    mask * no_overlap * self.q_value_bound * th.ones_like(current_novice_q_value)
                )

            # Compute Huber loss (less sensitive to outliers)
            loss_td = F.smooth_l1_loss(current_behavior_q_values, target_q_values)

            #loss = loss_td.mean() + pvp_loss.mean()
            if self.gamma_classifier >= 0:
                loss = loss_td.mean() + pvp_loss.mean()
            else:
                loss = pvp_loss.mean()
            
            # BC loss
            lp = torch.distributions.Categorical(logits=current_q_values).log_prob(replay_data.actions_behavior.flatten())
            masked_lp = (mask.flatten() * lp.flatten()).sum() / (mask.sum() + 1e-8)
            bc_loss = -lp.mean()
            
            stat_recorder["c_loss"].append(loss.item())
            stat_recorder["c_loss_bc"].append(bc_loss.item())

            # Optimize the policy
            self.classifier.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)
            self.classifier.optimizer.step()

            stat_recorder["c_value_behavior"].append(current_behavior_q_values.mean().item())
            stat_recorder["c_value_novice"].append(current_novice_q_value.mean().item())
        
        if n_upd > 0:
            with th.no_grad():
                if self.replay_buffer.pos >= 32:
                    replay_data = self.replay_buffer.sample(int(batch_size), env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                else:
                    replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env, discard_rgb=discard_rgb, return_features=return_features)
                
                current_q_values = self.classifier.q_net(replay_data.observations)
                current_novice_q_value_method1 = th.gather(current_q_values, dim=1, index=replay_data.actions_novice.long())
            self.switch2human_thresh = th.quantile(current_novice_q_value_method1, self.thr_classifier).item()
            
        self.delay += 1
        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

        self.logger.record("train/agent_buffer_size", self.replay_buffer.get_buffer_size())
        self.logger.record("train/human_buffer_size", self.human_data_buffer.get_buffer_size())

        for key, values in stat_recorder.items():
            self.logger.record("train/{}".format(key), np.mean(values))

        # Compute entropy (copied from RLlib TF dist)
        self.logger.record("train/entropy", np.mean(entropies))

    def _setup_model(self) -> None:
        super(PVPDQN, self)._setup_model()
        self.human_data_buffer = HACOReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs,
            optimize_memory_usage=self.optimize_memory_usage,
            **self.replay_buffer_kwargs
        )
        # self.human_data_buffer = self.replay_buffer

    def _store_transition(
            self,
            replay_buffer: ReplayBuffer,
            buffer_action: np.ndarray,
            new_obs: Union[np.ndarray, Dict[str, np.ndarray]],
            reward: np.ndarray,
            dones: np.ndarray,
            infos: List[Dict[str, Any]],
            backbone_features=None
    ) -> None:
        if infos[0]["takeover"] or infos[0]["takeover_start"]:
            replay_buffer = self.human_data_buffer
        super(PVPDQN, self)._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos, backbone_features=backbone_features)

    def save(
            self,
            path: Union[str, pathlib.Path, io.BufferedIOBase],
            exclude: Optional[Iterable[str]] = None,
            include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()
        # print(data)
        del data["replay_buffer"]
        del data["human_data_buffer"]
        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)
    
    def save_replay_buffer(
        self, path_human: Union[str, pathlib.Path, io.BufferedIOBase], path_replay: Union[str, pathlib.Path,
                                                                                          io.BufferedIOBase]
    ) -> None:
        save_to_pkl(path_human, self.human_data_buffer, self.verbose)
        super(PVPDQN, self).save_replay_buffer(path_replay)

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
        super(PVPDQN, self).load_replay_buffer(path_replay, truncate_last_traj)
    
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
        buffer_save_timesteps: int = 200,
        save_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        save_buffer: bool = False,
        load_buffer: bool = False,
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
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

        callback.on_training_start(locals(), globals())

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
            if save_buffer and self.num_timesteps > 0 and self.num_timesteps % buffer_save_timesteps == 0:
                buffer_location_human = os.path.join(
                    save_path_human, "human_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                buffer_location_replay = os.path.join(
                    save_path_replay, "replay_buffer_" + str(self.num_timesteps) + ".pkl"
                )
                logger.info("Saving..." + str(buffer_location_human))
                logger.info("Saving..." + str(buffer_location_replay))
                self.save_replay_buffer(buffer_location_human, buffer_location_replay)

        callback.on_training_end()

        return self
    
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
        load_path_human: Union[str, pathlib.Path, io.BufferedIOBase] = "",
        load_path_replay: Union[str, pathlib.Path, io.BufferedIOBase] = "",
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
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
            
        callback.on_training_end()

        return self