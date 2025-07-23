"""
Training script for training aim in MiniGrid environment.
"""
import argparse
import os
from pathlib import Path

# import gymnasium as gym
import torch

from aim.experiments.minigrid.minigrid_env import MiniGridMultiRoomN4S16, wrap_minigrid_env
from aim.experiments.minigrid.minigrid_model import MinigridCNN
from aim.aim_discrete import AIM_Discrete
from aim.sb3.common.callbacks import CallbackList, CheckpointCallback
from aim.sb3.common.monitor import Monitor
from aim.sb3.common.wandb_callback import WandbCallback
from aim.sb3.dqn.policies import CnnPolicy
from aim.utils.shared_control_monitor import SharedControlMonitor
from aim.utils.utils import get_time_str

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="AIM_minigrid", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--noisy_expert", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="", help="The team name for wandb.")
    parser.add_argument("--delta", type=float, default=0.1)
    args = parser.parse_args()

    # ===== Set up some arguments =====
    use_fake_human_with_failure = args.noisy_expert
    experiment_batch_name = "{}_{}".format(args.exp_name, "fourroomlarge")
    seed = args.seed
    trial_name = "{}_{}_{}".format("AIM", seed, get_time_str())
    free_level = 1 - args.delta

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    eval_log_dir = trial_dir / "evaluations"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(),

        # Algorithm config
        algo=dict(
            policy=CnnPolicy,
            policy_kwargs=dict(features_extractor_class=MinigridCNN, activation_fn=torch.nn.Tanh, net_arch=[
                64,
            ]),

            # === aim setting ===
            replay_buffer_kwargs=dict(discard_reward=True),
            exploration_fraction=0.0,
            exploration_initial_eps=0.0,
            exploration_final_eps=0.0,
            env=None,
            optimize_memory_usage=True,
            buffer_size=10_000,
            learning_rate=1e-4,

            # === New hypers ===
            learning_starts=10,
            batch_size=200,
            train_freq=(1, 'step'),
            tau=0.005,
            target_update_interval=1,
            gradient_steps=32,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
            init_bc_steps = 400,
            thr_classifier = free_level,
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )

    # ===== Setup the training environment =====
    env_class = MiniGridMultiRoomN4S16

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # env = wrap_minigrid_env(env_class, enable_takeover=True)
    env, unwrapped_env = wrap_minigrid_env(
        env_class, enable_takeover=False, use_fake_human=True, use_fake_human_robotgate=True, use_fake_human_with_failure=use_fake_human_with_failure
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    env = Monitor(env=env, filename=str(trial_dir))

    # TODO: Remove this for fakehuman experiment.
    # env = SharedControlMonitor(env=env, folder=trial_dir / "data", prefix=trial_name, save_freq=100)

    train_env = env

    # ===== Also build the eval env =====
    def _make_eval_env():
        env, _ = wrap_minigrid_env(env_class, enable_takeover=False)
        env = Monitor(env=env, filename=str(trial_dir))
        return env

    eval_env = _make_eval_env()
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Setup the callbacks =====
    save_freq = 100  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=1, save_freq=save_freq, save_path=str(trial_dir / "models"))
    ]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                trial_name=trial_name,
                exp_name=experiment_batch_name,
                team_name=team_name,
                project_name=project_name,
                config=config
            )
        )
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = AIM_Discrete(**config["algo"])
    import torch as th
    from aim.sb3.common.utils import get_schedule_fn
    classifier = CnnPolicy(model.observation_space,
                            model.action_space,
                            get_schedule_fn(1e-4),
                            features_extractor_class=MinigridCNN, 
                            activation_fn=th.nn.Tanh,
                            net_arch=[
                                64,
                            ])
    classifier = classifier.to("cuda")
    classifier.set_training_mode(True)
    model.classifier = classifier
    unwrapped_env.classifier = classifier
    unwrapped_env.model = model

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=2000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=20,
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
