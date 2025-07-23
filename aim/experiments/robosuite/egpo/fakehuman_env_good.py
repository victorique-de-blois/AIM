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
import numpy as np
import sys
import time

class CustomWrapper(gym.Env):
    last_takeover = None
    last_obs = None
    expert = None
    takeover = None
    last_turn = None
    from collections import deque 
    takeover_recorder = deque(maxlen=2000)
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    agent_action = None
    total_reward = 0
    t = 0
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
                "switch_to_expert": 0.2,
                "eval": False,
                "MAX_EP_LEN": 300,
                })
        self.config.update(config)
        # if self.config["eval"]:
        #     self.config["use_render"] = False
        self._render = self.config["use_render"]
    
    def obs_to_tensor(self, obs):
        obs_tensor = torch.as_tensor(obs).to("cuda")
        return obs_tensor
    
    def expert_act(self, o):
        obj_pos, obj_quat = o[:3], o[3:7]
        rel_quat = o[10:14]
        eef_pos, eef_quat = o[32:35], o[35:39]
        a = np.zeros(7)

        pose = pose2mat((obj_pos, obj_quat))
        grasp_point = (pose @ np.array([0.06, 0, 0, 1]))[:-1]

        if self.gripper_closed and np.linalg.norm(grasp_point - eef_pos) > 0.02:
            # open and lift gripper if it's not holding anything.
            a[-1] = -1.
            a[2] = 1.0
            return a

        if not self.gripper_closed and np.linalg.norm(grasp_point[:2] - eef_pos[:2]) > 0.005:
            # move gripper to be aligned with washer handle.
            a[-1] = -1.
            a[0:2] = 10 * (grasp_point[:2] - eef_pos[:2])
            self.last_turn = None
            return a

        if not self.gripper_closed and abs(rel_quat[0] + 1) > 0.01 and abs(rel_quat[1] + 1) > 0.01:
            # rotate gripper to be perpendicular to the washer.
            a[-1] = -1.
            if abs(obj_pos[2] - eef_pos[2]) < 0.02:
                a[2] = -30 * (obj_pos[2] - eef_pos[2])
            if self.last_turn:
                a[5] = self.last_turn
            elif abs(rel_quat[0] + 1) < abs(rel_quat[1] + 1): # rotate CW
                a[5] = -0.3
                self.last_turn = -0.3
            else: # rotate CCW
                a[5] = 0.3
                self.last_turn = 0.3
            return a

        if not self.gripper_closed and abs(obj_pos[2] - eef_pos[2]) > 0.0075:
            # move gripper to the height of the washer.
            a[-1] = -1.
            a[2] = 30 * (obj_pos[2] - eef_pos[2])
            return a

        if not self.gripper_closed:
            # grasp washer.
            a[-1] = 1.
            return a

        cylinder_pos = np.array([0.22690132, -0.10067187, 1.0])
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005 and abs(cylinder_pos[2] - eef_pos[2]) > 0.01:
            # move washer to correct height.
            a[-1] = 1.
            target_height = 1.0
            a[2] = 50 * (cylinder_pos[2] - eef_pos[2])
            return a
        
        if np.linalg.norm(cylinder_pos[:2] - obj_pos[:2]) > 0.005:
            # center above the cylinder.
            a[-1] = 1.
            a[0:2] = 50 * (cylinder_pos[:2] - obj_pos[:2])
            return a

        # lower washer down the cylinder.
        a[-1] = 1.
        a[2] = 50 * (0.83 - eef_pos[2])
        return a
    
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
        self.last_takeover = None
        self.takeover = None
        self.t = 0
        return r

    def get_state(self):
        obj = self._env.robots[0].controller
        np_array_attributes = {}
        for attr_name in dir(obj):
            # Skip private or special attributes (e.g., __init__, __class__)
            if attr_name.startswith("_"):
                continue
            
            # Get the attribute value
            attr_value = getattr(obj, attr_name)
            
            # Check if the attribute is a NumPy array
            if isinstance(attr_value, np.ndarray):
                np_array_attributes[attr_name] = attr_value
        
        return copy.deepcopy(np_array_attributes)

    def set_state(self, state):
        obj = self._env.robots[0].controller
        for attr_name, attr_value in state.items():
            # Set the NumPy array attribute
            try:
                setattr(obj, attr_name, attr_value)
            except:
                pass
                # TODO: torque_compensation is not settable

    def predict_agent_future_trajectory(self, current_obs, n_steps, action_behavior = None, return_all_states = False):
        info = dict()
        saved_state = copy.deepcopy(self._env.sim.get_state())
        
        controller_state = self.get_state()
        grip = self._env.robots[0].gripper.current_action[0]
        
        last_turn = self.last_turn
        traj = []
        obs = current_obs
        total_reward = 0
        success = False
        total_action_diff = 0
        for step in range(n_steps):
            action = action_behavior
            if action_behavior is None:
                action = self.agent_action
                if hasattr(self, "model"):
                     action, _ = self.model.policy.predict(obs, deterministic=True)
            action_expert = self.expert_act(obs)
            action_expert = np.clip(action_expert, -1, 1)
            action_diff = np.linalg.norm(action - action_expert)
            total_action_diff += action_diff
            step_reward = 0
            o, r, d, i = self.env.step(action)
            # self.render()
            step_reward += r
            settle_action = np.zeros(7)
            settle_action[-1] = action[-1]
            for _ in range(2):
                o, r, d, i = self.env.step(settle_action)
                # self.render()
                step_reward += r
            d = self._check_success()
            total_reward += step_reward
            traj.append({
                "obs": obs.copy(),
                "action": action.copy(),
                "reward": step_reward,
                "next_obs": o.copy(),
                "done": d,
                "action_expert": action_expert.copy(),
            })
            obs = o.copy()
            if d:
                success = True

        mean_action_diff = total_action_diff / n_steps
        self.last_turn = last_turn
        self._env.sim.set_state(saved_state)
        self.set_state(controller_state)
        self._env.robots[0].gripper.current_action[0] = grip
        
        info["success"] = success
        info["total_reward"] = total_reward
        info["mean_action_diff"] = mean_action_diff
        return traj, info
        
    def decide_takeover(self, obs, future_steps_predict):
        predicted_traj_real, info_real = self.predict_agent_future_trajectory(obs, future_steps_predict)
        #TODO: return other objectives. current: mean action difference
        return info_real["mean_action_diff"] > self.config["switch_to_expert"] and not self.config["eval"]
    
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
        
        if self.takeover == None or (self.total_steps % update_future_freq == 0):
            self.takeover = self.decide_takeover(self.last_obs, future_steps_predict)

        if self.takeover:
            #TODO: add to preference buffer
            action_ = expert_action

        step_reward = 0
        # time.sleep(1)
        o, r, d, i = self.env.step(action_)
        # time.sleep(1)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1
        self.total_reward += r
        step_reward += r
        self.render()
        settle_action = np.zeros(7)
        settle_action[-1] = action_[-1]
        for _ in range(2):
            o, r, d, i = self.env.step(settle_action)
            self.render()
            self.total_reward += r
            step_reward += r
        
        self.gripper_closed = self._env._check_grasp(gripper=self._env.robots[0].gripper, object_geoms=[g for g in self._env.nuts[self._env.nut_id].contact_geoms])
        self.last_obs = o
        i["raw_action"] = copy.copy(action_)
        i["step_reward"] = step_reward
        i["action_diff_new"] = np.mean((self.agent_action - expert_action) ** 2)
        i["takeover"] = i["takeover_cost"] = (self.takeover == True)
        i["gripper_closed"] = self.gripper_closed
        i["takeover_start"] = True if not self.last_takeover and self.takeover else False
        # condition = i["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        self.total_takeover_cost += i["takeover"]
        self.total_takeover_count += 1 if self.takeover else 0
        i["total_takeover_count"] = self.total_takeover_count
        i["total_takeover_cost"] = self.total_takeover_cost
        i["total_reward"] = self.total_reward
        self.t += 1
        d = (self.t >= self.config['MAX_EP_LEN']) or self._check_success()
        i["is_success"] = int(self._check_success())
        i["episode_length"] = self.t
        r += i["is_success"]
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
    robosuite_cfg = {'INPUT_DEVICE': input_device}
    # expert_pol = HardcodedPolicy(env).act
    
    num_episodes = 50
    i, failures = 0, 0
    np.random.seed(0)
    obs, act, rew = [], [], []
    act_limit = env.action_space.high[0]
    success_rate = 0
    while i < num_episodes:
        print('Episode #{}'.format(i))
        o, total_ret, d, t = env.reset(), 0, False, 0
        curr_obs, curr_act = [], []
        robosuite_cfg['INPUT_DEVICE'].start_control()
        while not d:
            # a = expert_pol(o)
            a = np.zeros(7) + 5
            o, r, d, _ = env.step(a)
            #print(r)
            t += 1
            success_rate += int(env._check_success())
        i += 1
        print("SR: ", success_rate / i, t)