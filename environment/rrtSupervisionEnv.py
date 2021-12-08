from .baseEnv import BaseEnv
from .rrt import RRTWrapper as RRT
import ray
import numpy as np
from .utils import perform_expert_actions
import torch
from policy.utils import preprocess_experiences
from itertools import chain

DEBUG_LOG = False
COMPUTE_RRT_ONLINE = False


@ray.remote
class RRTSupervisionEnv(BaseEnv):
    def __init__(self, env_config, training_config, gui, logger):
        super().__init__(
            env_config, training_config, gui, logger)
        # At the end of each episode, if arms fail to reach targets,
        # then invoke supervision to show the right way
        self.rrt_wrapper = None
        if COMPUTE_RRT_ONLINE:
            self.rrt_wrapper = RRT.remote(env_config, gui=gui)
        self.normal_mode()
        self.expert_root_dir = env_config['expert_root_dir']
        if self.expert_root_dir is None:
            print("[RRTSupervisionEnv] No expert loaded.")
        else:
            print("[RRTSupervisionEnv] Will be loading expert waypoints:",
                  self.expert_root_dir)

    def reset_supervision_stats(self):
        self.supervision_stats = {
            'supervision_count': 0,
            'supervision_failures': 0,
            'supervision_successes': 0
        }

    def setup(self, env_config, training_config, gui):
        super().setup(env_config, training_config, gui)
        self.expert_config = env_config['expert']

    def normal_mode(self):
        self.mode = 'normal'
        if DEBUG_LOG:
            print("Normal Mode")

    def supervision_mode(self):
        self.mode = 'supervision'
        self.collisions_in_supervision_mode = 0
        self.reset_supervision_stats()
        self.supervision_stats["supervision_count"] += 1
        if DEBUG_LOG:
            print("Supervision Mode")

    def is_supervision_mode(self):
        return self.mode == 'supervision'

    def mark_memories_for_delete(self):
        for m in self.ur5_episode_memories:
            m.mark_for_delete()

    def on_collision(self):
        super().on_collision()
        if self.is_supervision_mode():
            self.collisions_in_supervision_mode += 1

    def load_expert_episodes_for_task(self, task_id):
        expert_path = self.expert_root_dir + task_id + ".pt"
        try:
            expert_episodes = torch.load(expert_path)
        except Exception:
            return None
        return expert_episodes

    def save_expert_experiences_to_memory(self, expert_experiences):

        # each timesteps x num arms
        all_flat_observations, all_img_observations = preprocess_experiences(expert_experiences)

        for idx, (experience, flat_observations, img_observations) in enumerate(zip(expert_experiences, all_flat_observations, all_img_observations)):
            actions = experience[1]
            rewards = experience[3]
            self.terminate_episode = idx == (len(expert_experiences) - 1)
            self.observations = zip(flat_observations, img_observations)

            for flat_obs, img_obs, a, r, m in zip(
                    flat_observations,
                    img_observations,
                    actions,
                    rewards,
                    self.ur5_episode_memories):
                o = (flat_obs, img_obs)
                if idx > 0:
                    m.add_observation(o)
                m.add_action(a)
                m.add_rewards_and_termination(r, self.terminate_episode)
                m.add_value('next_observations', o)

    def on_episode_end(self):
        super().on_episode_end()
        if DEBUG_LOG:
            print('[RRTSupervisionEnv] on_episode_end')
        if self.is_supervision_mode():
            return None

        if self.finish_task_in_episode:
            # Robots succeeeded, don't need to show demonstration
            return None

        self.supervision_mode()
        self.reset_task()

        expert_experiences = None
        if self.expert_root_dir is not None:
            expert_experiences = self.load_expert_episodes_for_task(
                task_id=self.get_current_task().id)

        # if rrt_waypoints is None and self.rrt_wrapper is not None:
        #     rrt_waypoints = ray.get(self.rrt_wrapper.birrt_from_task.remote(
        #         self.get_current_task()))

        if expert_experiences is None or len(expert_experiences) == 0:
            self.mark_memories_for_delete()
            self.supervision_stats["supervision_failures"] += 1
            self.terminate_episode = True
            self.normal_mode()
            return None

        self.save_expert_experiences_to_memory(expert_experiences)

        if expert_experiences[-1][2]['reach_count'] == 2:
            self.supervision_stats["supervision_successes"] = 1
            self.send_memory_to_clusters()
        else:
            self.supervision_stats["supervision_failures"] = 1
            self.mark_memories_for_delete()
            self.terminate_episode = True

        assert self.is_supervision_mode()
        self.send_stats_to_logger()
        self.normal_mode()

        return expert_experiences


        # rv = perform_expert_actions(
        #     env=self,
        #     expert_waypoints=rrt_waypoints,
        #     expert_config=self.expert_config)
        #

        # TODO include this again after loading from expert episode data
        # once we actually save it.......
        # if self.collisions_in_supervision_mode > 0:
        #     self.supervision_stats["supervision_failures"] += 1
        # assert self.is_supervision_mode()
        # self.send_stats_to_logger()
        # self.normal_mode()
        #
        # # If failed to reach task, don't learn from it...
        # if self.supervision_stats["supervision_successes"] == 0:
        #     self.mark_memories_for_delete()
        #     self.terminate_episode = True
        # return rv

    def reset_task(self):
        assert self.is_supervision_mode()
        self.history = []
        self.current_step = 0

        self.setup_task()
        self.reset_stats()
        self.reset_memories()

        self.prev_eef_from_target = None
        self.episode_reward_sum = np.zeros(len(self.active_ur5s))
        self.observations = [self.preprocess_obs(o)
                             for o in self.get_observations()]

        for o, m in zip(self.observations, self.ur5_episode_memories):
            m.add_observation(o)
        self.terminate_episode = False
        return self.observations, self.ray_id

    # Call backs
    def on_all_ur5s_reach_target(self):
        self.terminate_episode = self.terminate_episode or \
            self.terminate_on_collectively_reach_target
        self.finish_task_in_episode = True
        if not self.is_supervision_mode():
            self.stats["collective_reach_count"] += 1
        else:
            self.supervision_stats["supervision_successes"] += 1
        if DEBUG_LOG:
            print('[RRTSupervisionEnv] All Reach Target!')

    # Overide baseEnv stats collection to only record stats under normal mode

    def on_step_simulation(self, curr, max, state):
        if not self.is_supervision_mode():
            super().on_step_simulation(curr, max, state)

    def should_get_next_task(self):
        if self.is_supervision_mode():
            return False
        else:
            return super().should_get_next_task()

    def on_target_reach(self, ur5, idx):
        ur5.on_touch_target()
        self.task_manager.targets_visuals[idx].touched()
        if not self.is_supervision_mode():
            self.stats["reached"][self.active_ur5s.index(ur5)] += 1

    def get_stats_to_log(self):
        if self.is_supervision_mode():
            return {
                'supervision_count':
                self.supervision_stats['supervision_count'],
                'supervision_failures':
                self.supervision_stats['supervision_failures'],
                'supervision_successes':
                self.supervision_stats['supervision_successes'],
            }
        else:
            retval = super().get_stats_to_log()
            retval['supervision_count'] = 0
            return retval
