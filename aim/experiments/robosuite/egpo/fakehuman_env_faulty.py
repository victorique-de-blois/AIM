#goal: add fake (imag) pos samples to neg trajs. see performance of algos
import copy
import math
import pathlib

import gymnasium as gym
import numpy as np
import torch
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import ckpt_path
from metadrive.policy.env_input_policy import EnvInputPolicy

from aim.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv

FOLDER_PATH = pathlib.Path(__file__).parent

logger = get_logger()

import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
# from thrifty.utils.hardcoded_nut_assembly import HardcodedPolicy
from robosuite.utils.transform_utils import pose2mat

from robosuite.wrappers import VisualizationWrapper
from robosuite.wrappers import GymWrapper
from robosuite.devices import Keyboard
from pyquaternion import Quaternion
import random
from robosuite import load_controller_config
from robosuite.utils.transform_utils import quat2axisangle
from robosuite.utils import RandomizationError

import numpy as np
import sys
import time
def _clip_delta(delta, max_step=0.015):
    norm_delta = np.linalg.norm(delta)

    if norm_delta < max_step:
        return delta
    return delta / norm_delta * max_step

class CustomWrapper(gym.Env):
    last_takeover = None
    last_obs = None
    expert = None
    takeover = False
    last_turn = None
    from collections import deque 
    takeover_recorder = deque(maxlen=2000)
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    agent_action = None
    total_reward = 0
    
    def __init__(self, env, unwrapped_env, config):
        self.env = env
        self._env = unwrapped_env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.gripper_closed = False
        self.viewer = env.viewer
        self.robots = env.robots
        self.config = dict({
                "use_discrete": False,
                "disable_expert": False,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": True,
                "future_steps_predict": 20,
                "update_future_freq": 10,
                "stop_img_samples": 3,
                "future_steps_preference": 1,
                "expert_noise": 0,
                "switch_to_expert": 0.05,
                })
        self.config.update(config)
        self._render = self.config["use_render"]
        self._g_tol = 5e-2 ** (0 + 1)
    
    def obs_to_tensor(self, obs):
        obs_tensor = torch.as_tensor(obs).to("cuda")
        return obs_tensor
    
    def _calculate_quat(self, angle):
        if "Sawyer" in self._env.robot_names:
            new_rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            return Quaternion(matrix=self._base_rot.dot(new_rot))
        return self._base_quat

    def get_handle_loc(self):
        if self._env.nut_id == 0:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('round-nut_handle_site')]
        elif self._env.nut_id == 1:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('SquareNut_handle_site')]
        else:
            handle_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('round-nut-3_handle_site')]
        handle_loc[2] = max(handle_loc[2], 0.01)
        return handle_loc

    def get_center_loc(self):
        if self._env.nut_id == 0:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('round-nut_center_site')]
        elif self._env.nut_id == 1:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('SquareNut_handle_site')]
        else:
            center_loc = self._env.sim.data.site_xpos[self._env.sim.model.site_name2id('round-nut-3_center_site')]
        center_loc[2] = max(center_loc[2], 0.005)
        return center_loc

    def _object_in_hand(self, obs):
        if np.linalg.norm(self.get_handle_loc() - obs[:3]) < 0.02:
            return True
        elif self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=[g for g in self._env.nuts[self._env.nut_id].contact_geoms]):
            return True
        return False

    def _get_target_pose(self, delta_pos, base_pos, quat, max_step=None):
        if max_step is None:
            max_step = self._default_speed

        delta_pos = _clip_delta(delta_pos, max_step)

        # if self.ranges.shape[0] == 7:
            # aa = np.concatenate(([quat.angle / np.pi], quat.axis))
        if False:    
            aa = [quat.angle / np.pi]
            if aa[0] < 0:
                aa[0] += 1
        else:
            quat = np.array([quat.x, quat.y, quat.z, quat.w])
            aa = quat2axisangle(quat)
        return np.clip(np.concatenate((delta_pos + base_pos, aa)), -1, 1)

    def expert_act(self, obs):
        status = 'start'
        if self._t == 0:
            self._start_grasp = -1
            self._finish_grasp = False

            y = -(self.get_handle_loc()[1] - obs[:3][1])
            x = self.get_handle_loc()[0] - obs[:3][0]

            angle = np.arctan2(y, x)
            self._target_quat = self._calculate_quat(angle)

        if self._start_grasp < 0 and self._t < 15:
            if np.linalg.norm(self.get_handle_loc() - obs[:3] + [0, 0, self._hover_delta]) < self._g_tol or self._t == 14:
                self._start_grasp = self._t

            quat_t = Quaternion.slerp(self._base_quat, self._target_quat, min(1, float(self._t) / 5))
            eef_pose = self._get_target_pose(
                self.get_handle_loc() - obs[:3] + [0, 0, self._hover_delta],
                obs[:3], quat_t)
            action = np.concatenate((eef_pose, [-1]))
            status = 'prepare_grasp'

        elif self._t < self._start_grasp + 45 and not self._finish_grasp:
            if not self._object_in_hand(obs):
                eef_pose = self._get_target_pose(
                    self.get_handle_loc() - obs[:3] - [0, 0, self._clearance],
                    obs[:3], self._target_quat)
                action = np.concatenate((eef_pose, [-1]))
                self.object_pos = obs['{}_pos'.format(self._object_name)]
                status = 'reaching_obj'
            else:
                eef_pose = self._get_target_pose(self.object_pos - obs[:3] + [0, 0, self._hover_delta],
                                                 obs[:3], self._target_quat)
                action = np.concatenate((eef_pose, [1]))
                if np.linalg.norm(self.object_pos - obs[:3] + [0, 0, self._hover_delta]) < self._g_tol:
                    self._finish_grasp = True
                status = 'obj_in_hand'

        elif np.linalg.norm(
                self._target_loc - self.get_center_loc()) > self._final_thresh and self._object_in_hand(obs):
            target = self._target_loc
            eef_pose = self._get_target_pose(target - self.get_center_loc(), obs[:3], self._target_quat)
            action = np.concatenate((eef_pose, [1]))
            status = 'moving'
        else:
            eef_pose = self._get_target_pose(np.zeros(3), obs[:3], self._target_quat)
            action = np.concatenate((eef_pose, [-1]))
            status = 'assembling'
        self._t += 1
        return action
    
    def reset(self):
        r = self.env.reset()
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = -1
        for _ in range(10):
            r, r2, r3, r4 = self.env.step(settle_action)
            self.render()
        self.gripper_closed = False
        self.last_obs = r
        self.last_takeover = False
        self._object_name = self._env.nuts[self._env.nut_id].name
        self._target_loc = np.array(self._env.sim.data.body_xpos[self._env.peg1_body_id]) + [0, 0, 0.115]
        self._clearance = 0.05

        # if "Sawyer" in self._env.robot_names:
        if True:
            self._obs_name = 'eef_pos'
            self._default_speed = 0.13
            self._final_thresh = 1e-2
            self._base_rot = np.array([[0, 1, 0.], [1, 0, 0.], [0., 0., -1.]])
            self._base_quat = Quaternion(matrix=self._base_rot)
        # elif "Panda" in self._env.robot_names:
        #     self._obs_name = 'eef_pos'
        #     self._default_speed = 0.13
        #     self._final_thresh = 1e-2
        #     self._base_rot = np.array([[1, 0, 0.], [0, -1, 0.], [0., 0., -1.]])
        #     self._base_quat = Quaternion(matrix=self._base_rot)
        # else:
        #     raise NotImplementedError

        self._t = 0
        self._intermediate_reached = False
        self._hover_delta = 0.15
        return r

    def step(self, action):
        # abstract 10 actions as 1 action
        # get rid of x/y rotation, which is unintuitive for remote teleop
        action_ = action.copy()
        action_[3] = 0.
        action_[4] = 0.
        
        self.agent_action = copy.copy(action_)
        self.last_takeover = self.takeover
        future_steps_predict = self.config["future_steps_predict"]
        update_future_freq = self.config["update_future_freq"]
        future_steps_preference = self.config["future_steps_preference"]
        expert_noise_bound = self.config["expert_noise"]
        
        expert_action = self.expert_act(self.last_obs)
        expert_action = np.clip(expert_action, -1, 1)
        enoise = np.random.randn(7) * expert_noise_bound
        expert_action = np.clip(enoise + expert_action, -1, 1)
        
        action_diff = np.linalg.norm(action_ - expert_action)
        if action_diff > self.config["switch_to_expert"]:
            #TODO: check whether set action_[3]/[4] to 0.0 affects the takeover decision
            action_ = expert_action
            self.takeover = True
        else:
            self.takeover = False
        
        step_reward = 0
        o, r, d, i = self.env.step(action_)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1
        self.total_reward += r
        step_reward += r
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action_[-1]
        for _ in range(10):
            o, r, d, i = self.env.step(settle_action)
            self.render()
            self.total_reward += r
            step_reward += r
        if action_[-1] > 0:
            self.gripper_closed = True
        else:
            self.gripper_closed = False
        self.last_obs = o
        i["step_reward"] = step_reward
        i["action_diff"] = action_diff
        i["takeover"] = self.takeover
        i["gripper_closed"] = self.gripper_closed
        i["takeover_start"] = True if not self.last_takeover and self.takeover else False
        # condition = i["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        self.total_takeover_cost += i["takeover"]
        self.total_takeover_count += 1 if self.takeover else 0
        i["total_takeover_count"] = self.total_takeover_count
        i["total_takeover_cost"] = self.total_takeover_cost
        i["total_reward"] = self.total_reward
        return o, r, d, i

    def _check_success(self):
        return self.env._check_success()

    def render(self):
        if self._render:
            self.env.render()

if __name__ == "__main__":
    
    render = True
    controller_config = load_controller_config(default_controller='OSC_POSE')
    config = {
        "env_name": "NutAssembly",
        "robots": "UR5e",
        "controller_configs": controller_config,
    }
    env = suite.make(
            **config,
            has_renderer=render,
            has_offscreen_renderer=False,
            render_camera="agentview",
            single_object_mode=2, # env has 1 nut instead of 2
            nut_type="round",
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            hard_reset=True,
            use_object_obs=True
        )
    unwrapped_env = env
    env = GymWrapper(env)
    env = VisualizationWrapper(env, indicator_configs=None)
    env = CustomWrapper(env, unwrapped_env, config=dict({"use_render": render}))

    arm_ = 'right'
    config_ = 'single-arm-opposed'
    input_device = Keyboard(pos_sensitivity=0.5, rot_sensitivity=3.0)
    if render:
        env.viewer.add_keypress_callback("any", input_device.on_press)
        env.viewer.add_keyup_callback("any", input_device.on_release)
        env.viewer.add_keyrepeat_callback("any", input_device.on_press)
    active_robot = env.robots[arm_ == 'left']
    robosuite_cfg = {'MAX_EP_LEN': 175, 'INPUT_DEVICE': input_device}
    # expert_pol = HardcodedPolicy(env).act
    
    num_episodes = 10
    i, failures = 0, 0
    np.random.seed(0)
    obs, act, rew = [], [], []
    act_limit = env.action_space.high[0]
    while i < num_episodes:
        print('Episode #{}'.format(i))
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        robosuite_cfg['INPUT_DEVICE'].start_control()
        while not d:
            # a = expert_pol(o)
            a = np.zeros(7)
            o, r, d, _ = env.step(a)
            print(r)
            d = (t >= robosuite_cfg['MAX_EP_LEN']) or env._check_success()
            r += int(env._check_success())
        i += 1