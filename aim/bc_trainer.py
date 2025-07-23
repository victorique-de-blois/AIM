import io
import logging
import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Union, Optional
import numpy as np
import torch as th
from torch.nn import functional as F
from aim.sb3.common.buffers import ReplayBuffer
from aim.sb3.common.save_util import load_from_pkl, save_to_pkl
from aim.sb3.common.type_aliases import GymEnv, MaybeCallback
from aim.sb3.common.utils import polyak_update
from aim.sb3.haco.haco_buffer import HACOReplayBuffer, concat_samples
from aim.sb3.td3.td3 import TD3
from aim.sb3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
logger = logging.getLogger(__name__)


class BCTrainer(TD3):
    def __init__(self, *args, **kwargs):
        """Please find the hyperparameters from original TD3"""
        super(BCTrainer, self).__init__(*args, **kwargs)


    def train(self, gradient_steps: int, batch_size: int, replay_data: ReplayBufferSamples = None) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer])

        stat_recorder = defaultdict(list)
        if replay_data == None:
            return stat_recorder

        lm = batch_size // self.env.num_envs
        for step in range(gradient_steps):
            self._n_updates += 1
            
            if replay_data is not None:
                # Compute actor loss
                a_pred = self.actor(replay_data.observations)
                loss_pi = th.mean((replay_data.actions_behavior - a_pred)**2)
                self.actor.optimizer.zero_grad()
                loss_pi.backward()
                self.actor.optimizer.step()
                stat_recorder["loss_pi"].append(loss_pi.item())

        return stat_recorder

