"""
Compared to original file:
1. use fakehumanenv
2. new config: free_level
3. buffer_size and total_timesteps set to 150_000
"""
import argparse
import os
from pathlib import Path
import uuid

from pvp.experiments.metadrive.egpo.fakehuman_env_ours import FakeHumanEnv
from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
from pvp.pvp_td3_ours_offline import PVPTD3ENS
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
        "--exp_name", default="pvp_metadrive_fakehuman", type=str, help="The name for this batch of experiments."
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--learning_starts", default=10, type=int)
    parser.add_argument("--save_freq", default=2500, type=int)
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--wandb", type=bool, default=True, help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="FIG3", help="The project name for wandb.")
    parser.add_argument("--wandb_team", type=str, default="victorique", help="The team name for wandb.")
    parser.add_argument("--log_dir", type=str, default="/home/caihy/pvp", help="Folder to store the logs.")
    parser.add_argument("--free_level", type=float, default=0.95)
    parser.add_argument("--bc_loss_weight", type=float, default=0.0)

    # parser.add_argument(
    #     "--intervention_start_stop_td", default=True, type=bool, help="Whether to use intervention_start_stop_td."
    # )

    parser.add_argument("--adaptive_batch_size", default="False", type=str)
    parser.add_argument("--only_bc_loss", default="False", type=str)
    parser.add_argument("--ckpt", default="", type=str)

    parser.add_argument("--toy_env", action="store_true", help="Whether to use a toy environment.")
    # parser.add_argument(
    #     "--device",
    #     required=True,
    #     choices=['wheel', 'gamepad', 'keyboard'],
    #     type=str,
    #     help="The control device, selected from [wheel, gamepad, keyboard]."
    # )
    parser.add_argument("--thr_classifier", type=float, default=0.95)
    parser.add_argument("--init_bc_steps", type=int, default=200)
    parser.add_argument("--thr_actdiff", type=float, default=0.4)
    
    args = parser.parse_args()

    # ===== Set up some arguments =====
    # control_device = args.device
    experiment_batch_name = "{}_offline".format("Thrifty")
    seed = args.seed
    trial_name = "{}_{}_{}".format(experiment_batch_name, seed, get_time_str())
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

    free_level = args.free_level
    thr_classifier = args.thr_classifier
    init_bc_steps = args.init_bc_steps
    thr_actdiff = args.thr_actdiff
    # ===== Setup the config =====
    config = dict(

        # Environment config
        env_config=dict(

            # Original real human exp env config:
            # use_render=True,  # Open the interface
            # manual_control=True,  # Allow receiving control signal from external device
            # controller=control_device,
            # window_size=(1600, 1100),

            # FakeHumanEnv config:
            free_level=free_level,
            thr_classifier=thr_classifier,
            init_bc_steps=init_bc_steps,
            thr_actdiff=thr_actdiff,
        ),

        # Algorithm config
        algo=dict(
            # intervention_start_stop_td=args.intervention_start_stop_td,
            adaptive_batch_size=args.adaptive_batch_size,
            bc_loss_weight=args.bc_loss_weight,
            only_bc_loss=args.only_bc_loss,
            add_bc_loss="True" if args.bc_loss_weight > 0.0 else "False",
            use_balance_sample=True,
            thr_classifier=thr_classifier,
            agent_data_ratio=1.0,
            policy=TD3Policy,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs=dict(
                discard_reward=True,  # We run in reward-free manner!
            ),
            policy_kwargs=dict(net_arch=[256, 256]),
            env=None,
            learning_rate=1e-4,
            q_value_bound=1,
            optimize_memory_usage=True,
            buffer_size=50_000,  # We only conduct experiment less than 50K steps
            learning_starts=args.learning_starts,  # The number of steps before
            batch_size=args.batch_size,  # Reduce the batch size for real-time copilot
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            action_noise=None,
            tensorboard_log=trial_dir,
            create_eval_env=False,
            verbose=2,
            #seed=seed,
            device="auto",
            num_instances=1,
            policy_delay=25,
            gradient_steps=5,
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
            # Here we set num_scenarios to 1, remove all traffic, and fix the map to be a very simple one.
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
    config["algo"]["env"] = train_env #make_vec_env(_make_train_env, n_envs=num_train_envs, vec_env_cls=SubprocVecEnv)
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
    model = PVPTD3ENS(**config["algo"])
    train_env.env.env.model = model

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    eval_freq, n_eval_episodes = 50 // num_train_envs, 50

    # ===== Launch training =====
    model.learn_offline(
        # training
        total_timesteps=50_000,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        # eval_env=None,
        # eval_freq=-1,
        # n_eval_episodes=2,
        # eval_log_path=None,

        # eval
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        save_buffer=False,
        load_buffer=True,
        load_path_human="/home/caihy/pvp/pvp/ens_human_buffer_2200.pkl",
    )
