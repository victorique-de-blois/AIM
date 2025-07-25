import copy
import os
import pickle

import numpy as np
from aim.utils.rllib_utils.callbacks import DrivingCallbacks
from aim.utils.rllib_utils.utils import initialize_ray
from ray import tune
from ray.tune import CLIReporter

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def train(
    trainer,
    config,
    stop,
    exp_name,
    num_seeds=1,
    num_gpus=0,
    test_mode=False,
    suffix="",
    checkpoint_freq=10,
    keep_checkpoints_num=None,
    start_seed=0,
    local_mode=False,
    save_pkl=True,
    custom_callback=None,
    max_failures=5,
    wandb_key_file=None,
    wandb_project=None,
    wandb_team="drivingforce",
    wandb_log_config=True,
    init_kws=None,
    **kwargs
):
    init_kws = init_kws or dict()
    # initialize ray
    if not os.environ.get("redis_password"):
        initialize_ray(test_mode=test_mode, local_mode=local_mode, num_gpus=num_gpus, **init_kws)
    else:
        password = os.environ.get("redis_password")
        assert os.environ.get("ip_head")
        print(
            "We detect redis_password ({}) exists in environment! So "
            "we will start a ray cluster!".format(password)
        )
        if num_gpus:
            print(
                "We are in cluster mode! So GPU specification is disable and"
                " should be done when submitting task to cluster! You are "
                "requiring {} GPU for each machine!".format(num_gpus)
            )
        initialize_ray(address=os.environ["ip_head"], test_mode=test_mode, redis_password=password, **init_kws)

    # prepare config
    used_config = {
        "seed": tune.grid_search([i * 100 + start_seed for i in range(num_seeds)]) if num_seeds is not None else None,
        "log_level": "DEBUG" if test_mode else "INFO",
        "callbacks": custom_callback if custom_callback else DrivingCallbacks,  # Must Have!
    }
    if custom_callback is False:
        used_config.pop("callbacks")
    if config:
        used_config.update(config)
    config = copy.deepcopy(used_config)

    if isinstance(trainer, str):
        trainer_name = trainer
    elif hasattr(trainer, "_name"):
        trainer_name = trainer._name
    else:
        trainer_name = trainer.__name__

    if not isinstance(stop, dict) and stop is not None:
        assert np.isscalar(stop)
        stop = {"timesteps_total": int(stop)}

    if keep_checkpoints_num is not None and not test_mode:
        assert isinstance(keep_checkpoints_num, int)
        kwargs["keep_checkpoints_num"] = keep_checkpoints_num
        kwargs["checkpoint_score_attr"] = "episode_reward_mean"

    if "verbose" not in kwargs:
        kwargs["verbose"] = 1 if not test_mode else 2

    # This functionality is not supported yet!
    metric_columns = CLIReporter.DEFAULT_COLUMNS.copy()
    progress_reporter = CLIReporter(metric_columns)
    progress_reporter.add_metric_column("success")
    progress_reporter.add_metric_column("crash")
    progress_reporter.add_metric_column("out")
    progress_reporter.add_metric_column("max_step")
    progress_reporter.add_metric_column("length")
    progress_reporter.add_metric_column("cost")
    progress_reporter.add_metric_column("takeover")
    progress_reporter.add_metric_column("route_completion")
    kwargs["progress_reporter"] = progress_reporter

    if wandb_key_file is not None:
        assert wandb_project is not None
    if wandb_project is not None:
        assert wandb_project is not None
        failed_wandb = False
        try:
            from aim.utils.rllib_utils.ray_wandb_callbacks import OurWandbLoggerCallback
        except Exception as e:
            # print("Please install wandb: pip install wandb")
            failed_wandb = True

        if failed_wandb:
            from ray.tune.logger import DEFAULT_LOGGERS
            from aim.utils.rllib_utils.ray_wandb_callbacks_ray100 import OurWandbLogger
            kwargs["loggers"] = DEFAULT_LOGGERS + (OurWandbLogger, )
            config["logger_config"] = {
                "wandb": {
                    "group": exp_name,
                    "exp_name": exp_name,
                    "entity": wandb_team,
                    "project": wandb_project,
                    "api_key_file": get_api_key_file(wandb_key_file),
                    "log_config": wandb_log_config,
                }
            }
        else:
            kwargs["callbacks"] = [
                OurWandbLoggerCallback(
                    exp_name=exp_name,
                    api_key_file=get_api_key_file(wandb_key_file),
                    project=wandb_project,
                    group=exp_name,
                    log_config=wandb_log_config,
                    entity=wandb_team
                )
            ]

    # start training
    analysis = tune.run(
        trainer,
        name=exp_name,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True if "checkpoint_at_end" not in kwargs else kwargs.pop("checkpoint_at_end"),
        stop=stop,
        config=config,
        max_failures=max_failures if not test_mode else 0,
        reuse_actors=False,
        local_dir=".",
        **kwargs
    )

    # save training progress as insurance
    if save_pkl:
        pkl_path = "{}-{}{}.pkl".format(exp_name, trainer_name, "" if not suffix else "-" + suffix)
        with open(pkl_path, "wb") as f:
            data = analysis.fetch_trial_dataframes()
            pickle.dump(data, f)
            print("Result is saved at: <{}>".format(pkl_path))
    return analysis
