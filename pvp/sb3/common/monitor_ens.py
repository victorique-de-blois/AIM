__all__ = ["Monitor", "get_monitor_files"]

import csv
import json
import os
import time
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pandas

from pvp.sb3.common.type_aliases import GymObs, GymStepReturn
from collections import defaultdict
import logging

logger = logging.getLogger(__file__)


class Monitor(gym.Wrapper):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param filename: the location to save a log file, can be None for no log
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    EXT = "monitor.csv"

    def __init__(
        self,
        env: gym.Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        # info_keywords: Tuple[str, ...] = (),
    ):
        super(Monitor, self).__init__(env=env)

        # PZH: Step the environment for once to understand the info keys.
        self.env.reset()
        o, r, d, i = self.env.step(self.env.action_space.sample())
        info_keywords = tuple(i.keys())
        reset_keywords = tuple(reset_keywords)
        ep_info_keywords = tuple("ep_" + k for k in info_keywords)
        record_keys = reset_keywords + info_keywords + ep_info_keywords
        logger.info("In Monitor, we auto detect the record keys: {}".format(record_keys))

        self.t_start = time.time()
        self.results_writer = None
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.metadata["info_keywords"] = self.info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_returns = []
        self.episode_lengths = []
        self.episode_times = []

        # PZH: Ours
        self.episode_infos = defaultdict(list)

        self.total_steps = 0
        self.current_reset_info = {}  # extra info about the current episode, that was passed in during reset()

    def reset(self, **kwargs) -> GymObs:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        for key in self.info_keywords:
            try:
                self.episode_infos[key].append(info[key])
            except:
                continue

        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
                ep_data = np.asarray(self.episode_infos[key])
                if ep_data.dtype == object:
                    pass
                else:
                    # Temporary workaround solution for accessing mean for non float/int
                    try:
                        ep_info["ep_{}".format(key)] = np.mean(ep_data)
                    except TypeError:
                        pass
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info
            self.episode_infos.clear()
        self.total_steps += 1
        return observation, reward, done, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super(Monitor, self).close()
        if self.results_writer is not None:
            self.results_writer.close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> List[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> List[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> List[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['env']
        return state

class LoadMonitorResultsError(Exception):
    """
    Raised when loading the monitor log fails.
    """

    pass


