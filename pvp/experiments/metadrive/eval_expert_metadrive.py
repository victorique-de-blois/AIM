"""
Training script for training PPO in MetaDrive Safety Env.
"""
import argparse
import os
from pathlib import Path

from pvp.sb3.common.callbacks import CallbackList, CheckpointCallback
from pvp.sb3.common.env_util import make_vec_env
from pvp.sb3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pvp.sb3.common.wandb_callback import WandbCallback
from pvp.sb3.ppo import PPO
from pvp.sb3.ppo.policies import ActorCriticPolicy
from pvp.utils.utils import get_time_str


def register_env(make_env_fn, env_name):
    from gym.envs.registration import register
    register(id=env_name, entry_point=make_env_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--seed", default=0, type=int, help="The random seed.")
    parser.add_argument("--ckpt", default=None, type=str, help="Path to previous checkpoint.")
    parser.add_argument("--debug", action="store_true", help="Set to True when debugging.")
    parser.add_argument("--wandb", type=bool, default=True, help="Set to True to upload stats to wandb.")
    parser.add_argument("--wandb_project", type=str, default="table1", help="The project name for wandb.")
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
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(),  uuid.uuid4().hex[:8])

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
            horizon=1500,
        ),
        num_train_envs=4,

        # ===== Environment =====
        eval_env_config=dict(
            use_render=False,  # Open the interface
            manual_control=False,  # Allow receiving control signal from external device
            start_seed=1000,
            horizon=1500,
        ),
        num_eval_envs=1,

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

    def _make_train_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        train_env = HumanInTheLoopEnv(config=train_env_config)
        train_env = Monitor(env=train_env, filename=str(trial_dir))
        return train_env

    train_env_name = "metadrive_train-v0"
    register_env(_make_train_env, train_env_name)
    train_env = make_vec_env(_make_train_env, n_envs=config["num_train_envs"], vec_env_cls=vec_env_cls)
    config["algo"]["env"] = train_env
    assert config["algo"]["env"] is not None

    # ===== Also build the eval env =====
    eval_env_config = config["eval_env_config"]

    def _make_eval_env():
        from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
        from pvp.sb3.common.monitor import Monitor
        eval_env = HumanInTheLoopEnv(config=eval_env_config)
        eval_env = Monitor(env=eval_env, filename=str(trial_dir))
        return eval_env

    eval_env_name = "metadrive_eval-v0"
    register_env(_make_eval_env, eval_env_name)
    eval_env = make_vec_env(_make_eval_env, n_envs=config["num_eval_envs"], vec_env_cls=vec_env_cls)

    # ===== Setup the callbacks =====
    save_freq = 10000  # Number of steps per model checkpoint
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
    model.learn(
        # training
        total_timesteps=0,
        callback=callbacks,
        reset_num_timesteps=True,

        # eval
        eval_env=eval_env,
        eval_freq=200,
        n_eval_episodes=10,
        eval_log_path=str(trial_dir),

        # logging
        tb_log_name=experiment_batch_name,
        log_interval=1,
        # save_buffer=False,
        # load_buffer=False,
    )

    if True:
        ckpt = "/home/caihy/pvp/pvp/experiments/metadrive/egpo/metadrive_pvp_20m_steps.zip"
        print(f"Loading checkpoint from {ckpt}!")
        from pvp.sb3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    from typing import Any, Callable, Dict, List, Optional, Union
    _is_success_buffer = []
    from collections import defaultdict
    from pvp.sb3.common.evaluation import evaluate_policy
    evaluations_info_buffer = defaultdict(list)
    def _log_success_callback( locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                _is_success_buffer.append(maybe_is_success)

            maybe_is_success2 = info.get("arrive_dest", None)
            if maybe_is_success2 is not None:
                _is_success_buffer.append(maybe_is_success2)

            assert (maybe_is_success is None) or (maybe_is_success2 is None), "We cannot have two success flags!"

            for k in ["episode_energy", "route_completion", "total_cost", "arrive_dest", "max_step", "out_of_road",
                      "crash"]:
                if k in info:
                    evaluations_info_buffer[k].append(info[k])

        if "raw_action" in info:
            evaluations_info_buffer["raw_action"].append(info["raw_action"])
    if True:
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=50,
                return_episode_rewards=True,
                callback=_log_success_callback,
            )

            print("Finish evaluating policy for {} episodes!".format(50))

            if False:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )
            import numpy as np
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            last_mean_reward = mean_reward

            if True:
                print(
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            model.logger.record("eval/mean_reward", float(mean_reward))
            model.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(_is_success_buffer) > 0:
                success_rate = np.mean(_is_success_buffer)
                print(f"Success rate: {100 * success_rate:.2f}%")
                model.logger.record("eval/success_rate", success_rate)

            for k, v in evaluations_info_buffer.items():
                model.logger.record("eval/{}".format(k), np.mean(np.asarray(v)))

            # Dump log so the evaluation results are printed with the correct timestep
            model.logger.record("time/total_timesteps", model.num_timesteps)
            try:
                import wandb
                wandb.log(model.logger.name_to_value, step=model.num_timesteps)
            except:
                pass
            model.logger.dump(model.num_timesteps)
