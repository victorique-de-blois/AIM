import argparse
import os
from pathlib import Path
import uuid

from pvp.experiments.metadrive.egpo.fakehuman_env_ours import FakeHumanEnv
from pvp.pvp_td3_ours import AIM
from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.monitor_ens import Monitor
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.haco import HACOReplayBuffer
from pvp.sb3.td3.policies import TD3Policy
from pvp.utils.shared_control_monitor import SharedControlMonitor
from pvp.utils.utils import get_time_str
from pvp.sb3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from pvp.sb3.common.env_util import make_vec_env
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", default="AIM", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", type=bool, default=True, help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="ICML0710", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="/home/caihy/aim", help="Folder to store the logs.")
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)
    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    parser.add_argument("--thr_classifier", type=float, default=0.95)
    parser.add_argument("--init_bc_steps", type=int, default=200)
    
    args = parser.parse_args()

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = args.exp_name
    seed = args.seed
    trial_name = "{}_{}_{}".format("AIM", seed, get_time_str())
    print("Trial name is set to: ", trial_name)

    use_wandb = args.wandb
    project_name = args.wandb_project
    team_name = args.wandb_team
    if not use_wandb:
        print("[WARNING] Please note that you are not using wandb right now!!!")

    log_dir = args.log_dir
    experiment_dir = Path(log_dir) / Path("runs") / experiment_batch_name

    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=False)  # Avoid overwritting old experiment
    print(f"We start logging training data into {trial_dir}")

    thr_classifier = args.thr_classifier
    init_bc_steps = args.init_bc_steps
    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(
            init_bc_steps=init_bc_steps,
        ),

        # Algorithm config
        algo=dict(
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            device="auto",
            num_instances=1,
            policy_delay=25,
            gradient_steps=5,
            thr_classifier=thr_classifier,
            init_bc_steps=init_bc_steps,
        ),

        # Experiment log
        exp_name=experiment_batch_name,
        seed=seed,
        use_wandb=use_wandb,
        trial_name=trial_name,
        log_dir=str(trial_dir)
    )
    if args.toy_env:
        config["env_config"].update(
            num_scenarios=1,
            traffic_density=0.0,
            map="COT"
        )

    def _make_train_env():
        train_env = FakeHumanEnv(config=config["env_config"], )
        config["algo"]["classifier"] = train_env.classifier
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        train_env = SharedControlMonitor(env=train_env, folder=trial_dir / "data", prefix=trial_name)
        return train_env
    
    num_train_envs = 1
    train_env = _make_train_env()
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    def _make_eval_env():
        eval_env_config = dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        )
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env = make_vec_env(_make_eval_env, n_envs=1, vec_env_cls=SubprocVecEnv)
    
    # ===== Setup the callbacks =====
    save_freq = args.save_freq // num_train_envs  # Number of steps per model checkpoint
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
    model = AIM(**config["algo"])
    train_env.env.env.model = model

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    eval_freq, n_eval_episodes = 10, 100

    # ===== Launch training =====
    model.learn(
        # training
        total_timesteps=2500,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=False,
    )
