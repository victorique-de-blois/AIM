import copy
import time
from collections import deque

import numpy as np
from metadrive.engine.core.onscreen_message import ScreenMessage
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.policy.manual_control_policy import TakeoverPolicyWithoutBrake
from metadrive.utils.math import safe_clip
from pvp.sb3.td3.policies import TD3Policy
from pvp.sb3.common.utils import get_schedule_fn
import torch as th
from pvp.sb3.common.utils import safe_mean
ScreenMessage.SCALE = 0.1

HUMAN_IN_THE_LOOP_ENV_CONFIG = {
    # Environment setting:
    "out_of_route_done": True,  # Raise done if out of route.
    "num_scenarios": 50,  # There are totally 50 possible maps.
    "start_seed": 100,  # We will use the map 100~150 as the default training environment.
    "traffic_density": 0.06,

    # Reward and cost setting:    "cost_to_reward": True,  # Cost will be negated and added to the reward. Useless in PVP.
    "cos_similarity": False,  # If True, the takeover cost will be the cos sim between a_h and a_n. Useless in PVP.

    # Set up the control device. Default to use keyboard with the pop-up interface.
    "manual_control": True,
    "agent_policy": TakeoverPolicyWithoutBrake,
    "controller": "keyboard",  # Selected from [keyboard, xbox, steering_wheel].
    "only_takeover_start_cost": False,  # If True, only return a cost when takeover starts. Useless in PVP.

    # Visualization
    "vehicle_config": {
        "show_dest_mark": True,  # Show the destination in a cube.
        "show_line_to_dest": True,  # Show the line to the destination.
        "show_line_to_navi_mark": True,  # Show the line to next navigation checkpoint.
    },
    "horizon": 1500,
}


class HumanInTheLoopEnv(SafeMetaDriveEnv):
    """
    Human-in-the-loop Env Wrapper for the Safety Env in MetaDrive.
    Add code for computing takeover cost and add information to the interface.
    """
    total_steps = 0
    total_takeover_cost = 0
    total_takeover_count = 0
    total_cost = 0
    total_switch = 0
    human_end_takeover = []
    agent_end_explore = []
    len_wo_switch = 0
    takeover = False
    last_t = False
    takeover_recorder = deque(maxlen=2000)
    agent_action = None
    in_pause = False
    start_time = time.time()
    warning = True
    total_miss = 0
    total_mode_changes = 0
    latency_agent = 3
    latency_human = 8
    history = deque(maxlen=10)
    lst_act_diff = []
    uncertainty = 0
    color_red = True
    classifier: TD3Policy
    
    def __init__(self, config):
        super(HumanInTheLoopEnv, self).__init__(config)
        self.classifier = TD3Policy(self.observation_space,
                    self.action_space,
                    get_schedule_fn(self.config["lr_classifier"]))
        self.classifier = self.classifier.to("cuda")
        self.classifier.set_training_mode(True)
    
    def compute_uncertainty(self, actions):
        th_obs = th.from_numpy(np.expand_dims(self.last_obs, 0)).to(self.classifier.device)
        th_actions = th.from_numpy(np.expand_dims(actions, 0)).to(self.classifier.device)
        unc = self.classifier.critic(th_obs, th_actions)[0].item()
        return unc
    
    def default_config(self):
        config = super(HumanInTheLoopEnv, self).default_config()
        config.update(HUMAN_IN_THE_LOOP_ENV_CONFIG, allow_add_new_key=True)
        config.update(
            {
                "init_bc_steps": 500,
                "lr_classifier": 1e-4,
                "thr_classifier_sw2agent": 0,
                "thr_classifier_sw2human": 0.1,
                "thr_actdiff": 0.04,
            }
        )
        return config

    def reset(self, *args, **kwargs):
        self.takeover = False
        self.last_t = False
        self.agent_action = None
        obs, info = super(HumanInTheLoopEnv, self).reset(*args, **kwargs)
        # The training code is for older version of gym, so we discard the additional info from the reset.
        self.last_obs = obs
        return obs
    
    def lasteq(self, dq: deque, n: int, a: bool):
        if len(dq) < n:
            return False
        import itertools
        arr = itertools.islice(dq, len(dq) - n, len(dq))
        return (bool)(all(x == a for x in arr))

    def _get_step_return(self, actions, engine_info):
        """Compute takeover cost here."""
        o, r, tm, tc, engine_info = super(HumanInTheLoopEnv, self)._get_step_return(actions, engine_info)
        self.last_obs = o
        d = tm or tc

        shared_control_policy = self.engine.get_policy(self.agent.id)
        last_t = self.takeover
        self.takeover = shared_control_policy.takeover if hasattr(shared_control_policy, "takeover") else False
        engine_info["takeover_start"] = True if not last_t and self.takeover else False
        engine_info["takeover"] = self.takeover
        
        self.last_t = last_t
        switch = (last_t != self.takeover)
        if not switch:
            self.len_wo_switch += 1
        else:
            if self.len_wo_switch > 0:
                if last_t:
                    self.human_end_takeover.append(self.len_wo_switch)
                else:
                    self.agent_end_explore.append(self.len_wo_switch)
            self.len_wo_switch = 1
        if d:
            if last_t:
                self.human_end_takeover.append(self.len_wo_switch)
            else:
                self.agent_end_explore.append(self.len_wo_switch)
        from pvp.sb3.common.utils import safe_mean
        engine_info["human_end_takeover"] = safe_mean(self.human_end_takeover)
        engine_info["agent_end_explore"] = safe_mean(self.agent_end_explore)
        engine_info["switch"] = switch
        self.total_switch += switch
        engine_info["total_switch"] = self.total_switch
        engine_info["total_mode_changes"] = self.total_mode_changes
        
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
        engine_info["total_cost"] = self.total_cost
        engine_info["mean_act_diff"] = safe_mean(self.lst_act_diff[-100:])
        engine_info["total_mode_changes"] = self.total_mode_changes
        engine_info["total_miss"] = self.total_miss
        engine_info["uncertainty"] = self.uncertainty
        # engine_info["total_cost_so_far"] = self.total_cost
        return o, r, d, engine_info

    def _is_out_of_road(self, vehicle):
        """Out of road condition"""
        ret = (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def step(self, actions):
        """Add additional information to the interface."""
        self.agent_action = copy.copy(actions)
        ret = super(HumanInTheLoopEnv, self).step(actions)
        
        last_warning = self.warning
        if self.takeover and not last_warning and not self.last_t:
            self.total_miss += 1
        
        # uncertainty = self.compute_uncertainty(actions)
        # if last_warning:
        #     thr_classifier = self.config["thr_classifier_sw2agent"]
        # else:
        #     thr_classifier = self.config["thr_classifier_sw2human"]
        # thr_actdiff = self.config["thr_actdiff"]
        if hasattr(self, "model") and hasattr(self.model, "trained"):
            var_actions = self.compute_uncertainty(actions)
            
            if not self.takeover:
                act_diff = 0
                decision = var_actions - self.model.switch2human_thresh
            else:
                act_diff = np.mean((np.array(ret[-1]["raw_action"]) - actions) ** 2)
                self.lst_act_diff.append(act_diff)
                decision = max(act_diff - self.config['thr_actdiff'], var_actions - self.model.switch2human_thresh)
        else:
            act_diff = 0
            var_actions = 0
            decision = 1
        
        self.history.append((bool)(decision > 0))
        if last_warning:
            self.warning = not self.lasteq(self.history, self.latency_human, False)
            #switch to "not warning" if all last self.latency_human steps have decision < 0
        else:
            self.warning = self.lasteq(self.history, self.latency_agent, True)
        if self.total_steps < self.config["init_bc_steps"]:
            self.warning = True
            #set warning = true if self.totaltimesteps < initbcsteps
        
        last_color_red = self.color_red
        self.color_red = self.warning
        self.total_mode_changes += (last_color_red != self.color_red)
        from panda3d.core import CardMaker, NodePath
        cm = CardMaker('rect')
        cm.set_frame(0, 0.2, 0, 0.1)
        self.engine.rect = NodePath(cm.generate())
        self.engine.rect.reparent_to(aspect2d)
        self.engine.rect.set_pos(-1, 0, 0.8)
        self.engine.rect.set_color((int)(self.color_red), 1 - (int)(self.color_red), 0, 1)  # RGBA

        if self.takeover:
            self.warning = True
        
        while self.in_pause:
            self.engine.taskMgr.step()

        self.takeover_recorder.append(self.takeover)
        # if self.config["use_render"]:  # and self.config["main_exp"]: #and not self.config["in_replay"]:
        #     super(HumanInTheLoopEnv, self).render(
        #         text={
        #             "Color Changes": self.total_mode_changes,
        #             "Total Miss": self.total_miss,
        #             "Total Cost": round(self.total_cost, 2),
        #             "Takeover Cost": round(self.total_takeover_cost, 2),
        #             "Takeover": "TAKEOVER" if self.takeover else "NO",
        #             "Total Step": self.total_steps,
        #             "Total Time": time.strftime("%M:%S", time.gmtime(time.time() - self.start_time)),
        #             "Takeover Rate": "{:.2f}%".format(np.mean(np.array(self.takeover_recorder) * 100)),
        #             #"Pause": "Press E",
        #             "Diff in Actions": act_diff,
        #             "Mean Act diff": round(safe_mean(self.lst_act_diff[-100:]), 2),
        #             "Agent Uncertainty": round(var_actions, 2),
        #         }
        #     )
        self.uncertainty = var_actions
        self.total_steps += 1

        self.total_takeover_count += 1 if self.takeover else 0
        ret[-1]["total_takeover_count"] = self.total_takeover_count

        return ret

    def stop(self):
        """Toggle pause."""
        self.in_pause = not self.in_pause

    def setup_engine(self):
        """Introduce additional key 'e' to the interface."""
        super(HumanInTheLoopEnv, self).setup_engine()
        self.engine.accept("e", self.stop)

    def get_takeover_cost(self, info):
        """Return the takeover cost when intervened."""
        if not self.config["cos_similarity"]:
            return 1
        takeover_action = safe_clip(np.array(info["raw_action"]), -1, 1)
        agent_action = safe_clip(np.array(self.agent_action), -1, 1)
        multiplier = (agent_action[0] * takeover_action[0] + agent_action[1] * takeover_action[1])
        divident = np.linalg.norm(takeover_action) * np.linalg.norm(agent_action)
        if divident < 1e-6:
            cos_dist = 1.0
        else:
            cos_dist = multiplier / divident
        return 1 - cos_dist


if __name__ == "__main__":
    env = HumanInTheLoopEnv({
        "manual_control": True,
        "use_render": True,
    })
    env.reset()
    while True:
        _, _, done, _ = env.step([0, 0])
        if done:
            env.reset()
