#visulize(save image step by step) of the dqn baseline ckpt for minigrid
import os
import time
import os.path as osp
import gym
import numpy as np
import pandas as pd
import torch
from gym_minigrid.wrappers import ImgObsWrapper
from aim.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from aim.sb3.dqn.policies import CnnPolicy
from aim.utils.print_dict_utils import pretty_print
from aim.aim.pvp_dqn.pvp_dqn import pvpDQN
from aim.training_script.atari.train_atari_dqn import DQN
from aim.training_script.minigrid.minigrid_env import MinigridWrapper
from aim.training_script.minigrid.minigrid_model import MinigridCNN

EVAL_ENV_START = 0


class AtariPolicyFunction:
    def __init__(self, ckpt_path, ckpt_index, env):
        self.algo = DQN(
            policy=CnnPolicy,
            policy_kwargs=dict(
                features_extractor_class=MinigridCNN,
                activation_fn=torch.nn.Tanh,
                net_arch=[
                    64,
                ]  # Remove FC in Q network
            ),
            env=env,
            optimize_memory_usage=True,

            # Hyper-parameters are collected from https://arxiv.org/pdf/1910.02078.pdf
            # MiniGrid specified parameters
            buffer_size=10_000,
            learning_rate=1e-4,
            exploration_fraction=0.30,  # Reach minimal exploration rate at 30% Total Steps
            exploration_final_eps=0.05,

            # === new set of hypers ===
            learning_starts=50,  # xxx: Original DQN has 100K warmup steps
            batch_size=256,  # Reduce the batch size for real-time copilot
            train_freq=1,
            tau=0.005,
            target_update_interval=1,
            gradient_steps=32,

            # tensorboard_log=log_dir,
            create_eval_env=False,
            verbose=2,
            # seed=seed,
            device="auto",
        )

        self.algo.set_parameters(load_path_or_dict=ckpt_path + "/rl_model_{}_steps".format(ckpt_index))

    def __call__(self, o, deterministic=False):
        assert deterministic
        action, state = self.algo.predict(o, deterministic=deterministic)
        return action


def evaluate_atari_once(
    ckpt_path,
    ckpt_index,
    folder_name,
    use_render=False,
    num_ep_in_one_env=5,
    env_name="BreakoutNoFrameskip-v4",
):
    ckpt_name = "checkpoint_{}".format(ckpt_index)
    # ===== Evaluate populations =====
    os.makedirs("evaluate_results", exist_ok=True)
    saved_results = []

    from aim.sb3.common.monitor import Monitor
    from gym.wrappers.time_limit import TimeLimit

    eval_log_dir = osp.join(ckpt_path, "evaluations")

    def _make_eval_env():
        env = gym.make(env_name)
        env = MinigridWrapper(env, enable_render=False, enable_human=False)
        env = Monitor(env=env, filename=eval_log_dir)
        env = ImgObsWrapper(env)
        env.seed(9)
        return env

    # eval_env = _make_eval_env()
    eval_env = VecFrameStack(DummyVecEnv([_make_eval_env]), n_stack=1)

    # Setup policy
    # try:
    policy_function = AtariPolicyFunction(ckpt_path, ckpt_index, eval_env)
    # except FileNotFoundError:
    #     print("We failed to load: ", ckpt_path)
    #     return None

    os.makedirs(folder_name, exist_ok=True)

    # Setup environment

    need_break = False
    # start = time.time()
    # last_time = time.time()
    ep_count = 0
    step_count = [0 for _ in range(eval_env.num_envs)]
    # rewards = 0
    # ep_times = []

    # env_index = 0
    o = eval_env.reset()

    # num_ep_in = 0
    from PIL import Image
    save_img = eval_env.render()
    curr_name = "minigridframe_{}.png".format('{0:03}'.format(step_count[0]))
    img = Image.fromarray(save_img)
    img.save(os.path.join(folder_name, curr_name))
    step_count[0] = step_count[0] + 1
    while not need_break:
        # INPUT: [batch_size, obs_dim] or [obs_dim, ] array.
        # OUTPUT: [batch_size, act_dim] !! This is important!
        action = policy_function(o, deterministic=True)

        # Step the environment
        o, _, dones, infos = eval_env.step(action)
        # rewards += r[0]

        # info = info[0]
        # d = d[0]

        for env_id, info in enumerate(infos):

            step_count[env_id] += 1

            if use_render:
                from PIL import Image
                save_img = eval_env.render()
                curr_name = "minigridframe_{}.png".format('{0:03}'.format(step_count[0]))
                img = Image.fromarray(save_img)
                img.save(os.path.join(folder_name, curr_name))

            if step_count[env_id] % 1000 == 0:
                print("Step {}, Episode {} ({})".format(step_count[env_id], ep_count, num_ep_in_one_env))

            if "episode" in info:
                print("Episode finished!")
                need_break = True
    return True


if __name__ == '__main__':
    index_x = []
    success_y = []
    success_rate = evaluate_atari_once(
        # ckpt_path=os.path.expanduser("/home/xxx/nvme/iclr_ckpt/minigrid_4room_baseline/"),
        # ckpt_index=50000,
        # folder_name="/home/xxx/nvme/iclr-visual/minigrid-4room-dqn",
        # ckpt_path=os.path.expanduser("/home/xxx/nvme/iclr_ckpt/minigrid_emptyroom_baseline/"),
        # ckpt_index=50000,
        # folder_name="/home/xxx/nvme/iclr-visual/minigrid-emptyroom-dqn5w",
        # ckpt_path=os.path.expanduser("/home/xxx/nvme/iclr_ckpt/minigrid_emptyroom_baseline/"),
        # ckpt_index=10000,
        # folder_name="/home/xxx/nvme/iclr-visual/minigrid-emptyroom-dqn1w",
        ckpt_path=os.path.expanduser("/home/xxx/nvme/iclr_ckpt/minigrid_2room_baseline/"),
        ckpt_index=50000,
        folder_name="/home/xxx/nvme/iclr-visual/minigrid-2room-dqnnew1",
        use_render=True,
        num_ep_in_one_env=50,
        env_name="MiniGrid-MultiRoom-N2-S4-v0",
        # env_name="MiniGrid-Empty-Random-6x6-v0",
        # env_name="MiniGrid-FourRooms-v0"
    )
