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
        
        max_pos = self.human_data_buffer.pos
        self.human_data_buffer.pos = min(max_pos,timestep)
        
        stat_recorders["wall_steps"].append(timestep)
        
        if True:
            self.actor.reset_parameters()
            self.trained = True


            ##first training of student policies: bc
            import tqdm
            for _ in tqdm.trange(2000, desc="Gradient Steps"):
                idx = np.random.randint(200, min(max_pos,timestep), size=batch_size)
                replay_data = self.human_data_buffer._get_samples(idx, env=self._vec_normalize_env)
                #replay_data = self.human_data_buffer.sample(int(batch_size), env=self._vec_normalize_env)
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
        self.human_data_buffer.pos = max_pos
        
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
        self.num_timesteps = 200
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
            if self.num_timesteps % 50 == 0:
                gradient_steps = self.gradient_steps
                assert gradient_steps > 0
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train_offline(batch_size=self.batch_size, gradient_steps=gradient_steps, timestep= self.num_timesteps)
            
        callback.on_training_end()
