import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
#from mj_envs.utils.quatmath import quat2euler, euler2quat
from mj_allegro_envs.allegro_hand_manipulation_suite import rotations

from mujoco_py import MjViewer
import os
import copy 
import pdb

ADD_BONUS_REWARDS = True

class BottleCapEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.cap_length = 1.0
        self.tar_length = 1.0
        self.desired_orien = np.array([0, 0, 3.14])


        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/bottlecap.xml', 5)

        utils.EzPickle.__init__(self)
        self.hand_bid = self.sim.model.body_name2id('robot0:hand')
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.obj_bid = self.sim.model.body_name2id('Object')
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_top')
        self.obj_b_sid = self.sim.model.site_name2id('object_bottom')
        self.tar_t_sid = self.sim.model.site_name2id('target_top')
        self.tar_b_sid = self.sim.model.site_name2id('target_bottom')

        self.goal = np.concatenate([self.data.site_xpos[self.eps_ball_sid].ravel(),self.desired_orien])
        self.cap_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1]) *2.57
        self.action_space.low  = -1.57 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        # self.model.body_quat[self.target_obj_bid] = euler2quat(np.array([3.14,1.5,0]))
        

    def step(self, a):
#        np.clip(a, self.action_space.low, self.action_space.high)
        a = np.clip(a,-1.57,2.57)
        orien_thresh = 0.709
        # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos  = self.data.body_xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.cap_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        
        # pos cost
        dist = np.linalg.norm(obj_pos-desired_loc)
        reward = -dist
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)
        reward += orien_similarity

        if ADD_BONUS_REWARDS:
            # bonus for being close to desired orientation
            if dist < 0.075 and orien_similarity > orien_thresh:
                reward += 10
            if dist < 0.075 and orien_similarity > 0.95:
                reward += 50

        # penalty for dropping the cap
        done = False
        if obj_pos[2] < 0.01:
            reward -= 100

        goal_achieved = True if (dist < 0.075 and orien_similarity > orien_thresh) else False        
        #print('orien_similarity: ' + str(orien_similarity))
        #print('dist' + str(dist))
        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def compute_reward(self, achieved_goal, goal, info):
        reward = np.ones(achieved_goal.shape[0])
        orien_thresh = 0.709
        for b in range(achieved_goal.shape[0]):
            obj_pos = achieved_goal[b][:3]
            desired_loc = goal[b][:3]
            obj_orien = achieved_goal[b][-4:]
            desired_orien = achieved_goal[b][-4:]
            dist = np.linalg.norm(obj_pos-desired_loc)

            reward[b] = -dist
            # orien cost
            orien_similarity = np.dot(obj_orien, desired_orien)
            reward[b] += orien_similarity

            if ADD_BONUS_REWARDS:
                # bonus for being close to desired orientation
                if dist < 0.075 and orien_similarity > orien_thresh:
                    reward[b] += 10
                if dist < 0.075 and orien_similarity > 0.95:
                    reward[b] += 50

            # penalty for dropping the cap
            if obj_pos[2] < 0.01:
                reward[b] -= 100
        return reward

    def get_achieved_goal(self):
        orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.cap_length
        pos  = self.data.body_xpos[self.obj_bid].ravel()
        return np.concatenate([pos,orien])

    def set_target_orien(self):
        self.model.body_quat[self.target_obj_bid] = rotations.euler2quat(self.desired_orien)

    def positon_hand(self):
        ''' Flips hand and places it closer to the table '''
        desired_hand_orien = np.array([3.14/2, 0, 4.71])
        self.model.body_quat[self.hand_bid] = rotations.euler2quat(desired_hand_orien)
        self.model.body_pos[self.hand_bid] = [1, 1.25, 0.09]

    def get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.body_xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.cap_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])/self.tar_length
        return np.concatenate([qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien,
                               obj_pos-desired_pos, obj_orien-desired_orien])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.set_target_orien()
        self.positon_hand()
        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.set_state(qp, qv)
        self.model.body_quat[self.target_obj_bid] = rotations.euler2quat(desired_orien)
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if cap within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
