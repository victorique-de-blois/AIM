"""
Training script for training PPO in MetaDrive Safety Env.
"""
import argparse
import os
from pathlib import Path

from aim.sb3.common.callbacks import CallbackList, CheckpointCallback
from aim.sb3.common.env_util import make_vec_env
from aim.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from aim.sb3.common.wandb_callback import WandbCallback
from aim.sb3.ppo import PPO
from aim.sb3.ppo.policies import ActorCriticPolicy
from aim.utils.utils import get_time_str
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from aim.experiments.robosuite.egpo.fakehuman_env import CustomWrapper, GymWrapper

def register_env(make_env_fn, env_name):
    from gym.envs.registration import register
    register(id=env_name, entry_point=make_env_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--ckpt", default=None, type=str, help="Path to previous checkpoint.")
    parser.add_argument("--debug", action="store_true", help="Set to True when debugging.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="Wipe", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    args = parser.parse_args()

    # FIXME: Remove this in future.
    if args.wandb_team is None:
        args.wandb_team = "drivingforce"
    if args.wandb_project is None:
        args.wandb_project = "pvp2024"

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}".format(args.exp_name)
    seed = args.seed
    import uuid
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup the config =====
    config = dict(
        # ===== Environment =====
        env_config=dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),
        ),
        num_train_envs=10,

        # ===== Training =====
        algo=dict(
            policy=ActorCriticPolicy,
            n_steps=512,  # n_steps * n_envs = total_batch_size
            n_epochs=20,
            learning_rate=5e-5,
            batch_size=256,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=10.0,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            seed=seed,
            device="auto",
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )
    vec_env_cls = SubprocVecEnv
    if args.debug:
        config["num_train_envs"] = 1
        config["algo"]["n_steps"] = 64
        vec_env_cls = DummyVecEnv

    # ===== Setup the training environment =====
    train_env_config = config["env_config"]

    # ===== Also build the eval env =====
    def _make_eval_env():
        from aim.sb3.common.monitor import Monitor
        # render = config["env_config"]["use_render"]
        render = False
        controller_config = load_controller_config(default_controller='OSC_POSE')
        configr = {
            "env_name": "Wipe",
            "robots": "UR5e",
            "controller_configs": controller_config,
        }
        env = suite.make(
                **configr,
                has_renderer=render,
                has_offscreen_renderer=False,
                render_camera="agentview",
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
        env = CustomWrapper(env, unwrapped_env, config=config["env_config"])
        train_env = Monitor(env=env, filename=str(trial_dir))
        return train_env
    eval_env = make_vec_env(_make_eval_env, n_envs=5, vec_env_cls=vec_env_cls)
    
    # eval_env = SubprocVecEnv([_make_eval_env] * 1)
    def _make_train_env():
        from aim.sb3.common.monitor import Monitor
        render = config["env_config"]["use_render"]
        controller_config = load_controller_config(default_controller='OSC_POSE')
        configr = {
            "env_name": "Wipe",
            "robots": "UR5e",
            "controller_configs": controller_config,
        }
        env = suite.make(
                **configr,
                has_renderer=render,
                has_offscreen_renderer=False,
                render_camera="agentview",
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
        env = CustomWrapper(env, unwrapped_env, config=config["env_config"])
        train_env = Monitor(env=env, filename=str(trial_dir))
        return train_env

    train_env_name = "metadrive_train-v0"
    register_env(_make_train_env, train_env_name)
    train_env = make_vec_env(_make_train_env, n_envs=config["num_train_envs"], vec_env_cls=vec_env_cls)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None
    # ===== Setup the callbacks =====
    save_freq = 2500  # Number of steps per model checkpoint
    callbacks = [
        CheckpointCallback(name_prefix="rl_model", verbose=2, save_freq=save_freq, save_path=str(trial_dir / "models"))
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
    model = PPO(**config["algo"])

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from aim.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=1000_0000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=2500,
        n_eval_episodes=50,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        # save_buffer=False,
        # load_buffer=False,
    )
