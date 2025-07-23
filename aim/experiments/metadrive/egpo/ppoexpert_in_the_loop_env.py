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


def get_expert():
    from aim.sb3.common.save_util import load_from_zip_file
    from aim.sb3.ppo import PPO
    from aim.sb3.ppo.policies import ActorCriticPolicy

    train_env = HumanInTheLoopEnv(config={'manual_control': False, "use_render": False})

    # Initialize agent
    algo_config = dict(
        policy=ActorCriticPolicy,
        n_steps=1024,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        # tensorboard_log=trial_dir,
        create_eval_env=False,
        verbose=2,
        # seed=seed,
        device="auto",
        env=train_env
    )
    model = PPO(**algo_config)

    ckpt = FOLDER_PATH / "metadrive_ppo_20m_steps"

    print(f"Loading checkpoint from {ckpt}!")
    data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
    model.set_parameters(params, exact_match=False, device=model.device)
    print(f"Model is loaded from {ckpt}!")

    train_env.close()

    return model.policy


def obs_correction(obs):
    # due to coordinate correction, this observation should be reversed
    obs[15] = 1 - obs[15]
    obs[10] = 1 - obs[10]
    return obs


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def load():
    global _expert_weights
    if _expert_weights is None:
        _expert_weights = np.load(ckpt_path)
    return _expert_weights


_expert = get_expert()


class PPOExpertInTheLoopEnv(HumanInTheLoopEnv):
    last_takeover = None
    last_obs = None
    expert = None
    total_switch = 0
    total_wall_steps = 0
    total_miss = 0

    def __init__(self, config):
        self.unc = None
        super(PPOExpertInTheLoopEnv, self).__init__(config)
        if self.config["use_discrete"]:
            self._num_bins = 13
            self._grid = np.linspace(-1, 1, self._num_bins)
            self._actions = np.array(np.meshgrid(self._grid, self._grid)).T.reshape(-1, 2)

    @property
    def action_space(self) -> gym.Space:
        if self.config["use_discrete"]:
            return gym.spaces.Discrete(self._num_bins ** 2)
        else:
            return super(PPOExpertInTheLoopEnv, self).action_space

    def default_config(self):
        """Revert to use the RL policy (so no takeover signal will be issued from the human)"""
        config = super(PPOExpertInTheLoopEnv, self).default_config()
        config.update(
            {
                "use_discrete": False,
                "disable_expert": False,
                "agent_policy": EnvInputPolicy,
                "init_bc_steps": 200,
                "lr_classifier": 1e-4,
                "manual_control": False,
                "use_render": False,
                "expert_deterministic": False,
            }
        )
        return config

    def continuous_to_discrete(self, a):
        distances = np.linalg.norm(self._actions - a, axis=1)
        discrete_index = np.argmin(distances)
        return discrete_index

    def discrete_to_continuous(self, a):
        continuous_action = self._actions[a.astype(int)]
        return continuous_action

    def step(self, actions):
        """Compared to the original one, we call expert_action_prob here and implement a takeover function."""
        actions = np.asarray(actions).astype(np.float32)

        if self.config["use_discrete"]:
            actions = self.discrete_to_continuous(actions)

        self.agent_action = copy.copy(actions)
        self.last_takeover = self.takeover

        # ===== Get expert action and determine whether to take over! =====
        if self.config["disable_expert"]:
            pass

        else:
            if self.expert is None:
                global _expert
                self.expert = _expert
            last_obs, _ = self.expert.obs_to_tensor(self.last_obs)
            distribution = self.expert.get_distribution(last_obs)
            if self.config["expert_deterministic"]:
                expert_action = distribution.mode().detach().cpu().numpy()
            else:
                expert_action = distribution.sample().detach().cpu().numpy()
            expert_action = expert_action[0]

            if self.config["use_discrete"]:
                expert_action = self.continuous_to_discrete(expert_action)
                expert_action = self.discrete_to_continuous(expert_action)            
            
            if self.total_steps <= self.config['init_bc_steps']:
                self.takeover = True
                self.total_wall_steps += 1
            else:
                unc = self.compute_uncertainty(actions)
                self.takeover = False
                if not self.last_takeover:
                    if unc > self.model.switch2human_thresh:
                        self.takeover = True
                else:
                    if np.mean((actions - expert_action) ** 2) > self.model.switch2robot_thresh:
                        self.takeover = True
                
            if self.takeover:
                actions = expert_action
                self.total_wall_steps += (self.total_steps > self.config['init_bc_steps'])

        o, r, d, i = super(HumanInTheLoopEnv, self).step(actions)
        self.takeover_recorder.append(self.takeover)
        self.total_steps += 1

        if self.config["use_render"]:
            super(HumanInTheLoopEnv, self).render(
                text={
                    "Total Cost": round(self.total_cost, 2),
                    "Takeover Cost": round(self.total_takeover_cost, 2),
                    "Takeover": "TAKEOVER" if self.takeover else "NO",
                    "Total Step": self.total_steps,
                    "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
                    "Pause": "Press E",
                }
            )

        assert i["takeover"] == self.takeover

        if self.config["use_discrete"]:
            i["raw_action"] = self.continuous_to_discrete(i["raw_action"])

        return o, r, d, i

    def _get_step_return(self, actions, engine_info):
        """Compared to original one, here we don't call expert_policy, but directly get self.last_takeover."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc
        last_t = self.last_takeover
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        condition = engine_info["takeover_start"] if self.config["only_takeover_start_cost"] else self.takeover
        if not condition:
            engine_info["takeover_cost"] = 0
        else:
            cost = self.get_takeover_cost(engine_info)
            self.total_takeover_cost += cost
            engine_info["takeover_cost"] = cost
        engine_info["total_takeover_cost"] = self.total_takeover_cost
        engine_info["native_cost"] = engine_info["cost"]
        engine_info["episode_native_cost"] = self.episode_cost
        self.total_cost += engine_info["cost"]
        self.total_takeover_count += 1 if self.takeover else 0
        engine_info["total_takeover_count"] = self.total_takeover_count
        engine_info["total_cost"] = self.total_cost
        sw = (last_t != self.takeover)
        self.total_switch += sw
        engine_info["switch"] = sw
        engine_info["total_switch"] = self.total_switch
        return o, r, d, engine_info

    def _get_reset_return(self, reset_info):
        o, info = super(HumanInTheLoopEnv, self)._get_reset_return(reset_info)
        self.last_obs = o
        self.last_takeover = False
        return o, info


if __name__ == "__main__":
    env = PPOExpertInTheLoopEnv(dict(use_render=False))
    env.reset()
    while True:
        _, _, done, info = env.step([0, 1])
        if done:
            print(info)
            env.reset()
