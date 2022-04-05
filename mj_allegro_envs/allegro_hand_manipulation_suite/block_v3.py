import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

from mj_allegro_envs.allegro_hand_manipulation_suite import rotations
ADD_BONUS_REWARDS = True

class BlockEnvV3(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', random_start=0):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self.obj_bid = 0
        self.target_obj_bid = 0
        self.obj_did = 0
        self.eps_ball_sid = 0
        self.obj_b_sid = 0
        self.tar_b_sid = 0
        self.obj_t_sid = 0
        self.tar_t_sid = 0
        self.obj_length = 0.025
        self.tar_length = 0.025
        self.desired_loc = [1,1,1]
        self.desired_orien = [1,1,1]
        self.default_quat = np.array([0.7,0.7, 0.0,0.0])

        self.possible_quat = [[0.704,0.704,-0.062,-0.062], #-10
                              [0.696,0.696,-0.123,-0.123], #-20
                              [0.683,0.683,-0.183,-0.183], #-30
                              [0.664,0.664,-0.242,-0.242], #-40
                              [0.641,0.641,-0.299,-0.299], #-50
                              [0.612,0.612,-0.354,-0.354], #-60
                              [0.574,0.574,-0.406,-0.406], #-70
                              [0.542,0.542,-0.455,-0.455]] #-80

        self.mix_goals = 0.0
        self.quat_candidates = [np.array([0.7,0.7, 0.0,0.0])]
        #TODO: have this set by passed in values, for now set to False
        self.target_position = 'ignore'
        self.target_rotation = 'z'
        self.target_position_range = np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)])
        self.ignore_z_target_rotation = False
        self.distance_threshold = 100
        self.rotation_threshold = 0.45
        self.goal = np.zeros(7)
        self.max_goal = self.default_quat
        self.reward_type = reward_type
        self.random_start = random_start

        self.offset1 = (np.random.random() - 0.5) * self.random_start 
        self.offset2 = (np.random.random() - 0.5) * self.random_start
        print(str(self.offset1) + " and " + str(self.offset2))

        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/blockv3.xml', 5)
        utils.EzPickle.__init__(self)
        self.original_qpos = self.init_qpos.copy()

        #back and forth
        self.init_qpos[17] =  self.original_qpos[17] + self.offset1
        #across palm
        self.init_qpos[16] = self.original_qpos[16] +  self.offset2
        self.init_qpos[18] = .21

        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1]) * 2.0
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        
        self.obj_bid = self.sim.model.body_name2id("Object")
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.eps_ball_sid = self.sim.model.site_name2id('eps_ball')
        self.obj_t_sid = self.sim.model.site_name2id('object_face')
        self.obj_b_sid = self.sim.model.site_name2id('object_back')

        self.tar_t_sid = self.sim.model.site_name2id('target_face')
        self.tar_b_sid = self.sim.model.site_name2id('target_back')
        self.desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        #Manually set, not displayed (YET!!TODO)
        self.desired_orien = np.array([ 0.2062416, 0.97764687, -0.04087782])

    def increase_goal_difficulty(self, new_quat):
        self.max_goal = new_quat
        self.set_goal_quat(new_quat)

    def quat_from_angle_and_axis(self,angle, axis):
        angle = np.radians(angle)
        assert axis.shape == (3,)
        axis /= np.linalg.norm(axis)
        quat = np.concatenate([[np.cos(angle / 2.)], np.sin(angle / 2.) * axis])
        quat /= np.linalg.norm(quat)
        return quat

    def angle_from_quaternion(self,quat):
        qw,qx,qy,qz = quat
        angle = 2 * np.arccos(qw)
        angle = np.degrees(angle)
        x = qx / np.sqrt(1-qw*qw)
        y = qy / np.sqrt(1-qw*qw)
        z = qz / np.sqrt(1-qw*qw)
        return angle, np.array((x,y,z))

    def set_manual_offset(self, off1, off2):
        self.offset1 = off1
        self.offset2 = off2
        self.init_qpos[17] = self.original_qpos[17] + self.offset1
        self.init_qpos[16] += self.original_qpos[16] + self.offset2
        self.init_qpos[18] = .21

    def step(self, a):
        try:
            starting_up = False
            assert(len(a)==6)

        except:
            starting_up = True
        
        self.do_simulation(a, self.frame_skip)
        achv_goal = self.sim.data.get_joint_qpos('object:joint')
        
        info = {}
        reward = self.compute_reward(achv_goal, self.goal, info)
        if not self.is_on_palm():
             reward -= 100000
        done = False
        goal_achieved = self._is_success(achv_goal, self.goal)
        info = {
            'is_success': goal_achieved,
        }
        return self.get_obs(), reward, done, dict(goal_achieved=goal_achieved)

    def is_on_palm(self):
        self.sim.forward()
        cube_middle_pos = self.data.body_xpos[self.obj_bid].ravel()
        is_on_palm = (cube_middle_pos[2] > 0.04)
        return is_on_palm

    def get_qp_free_joint(self, qp):
        quat = np.zeros(4)
        tx = qp[16]
        ty = qp[17]
        tz = qp[18]
        rx = qp[19]
        ry = qp[20]
        rz = qp[21]
        cx = np.cos(rx * 0.5)
        cy = np.cos(ry * 0.5)
        cz = np.cos(rz * 0.5)
        sx = np.sin(rx * 0.5)
        sy = np.sin(ry * 0.5)
        sz = np.sin(rz * 0.5)
        
        quat[0] = cx * cy * cz + sx * sy * sz
        quat[1] = sx * cy * cz - cx * sy * sz
        quat[2] = cx * sy * cz + sx * cy * sz
        quat[3] = cx * cy * sz - sx * sy * cz
        return quat


    #returns the qpos, difference in position from desired positon and velocity
    def get_obs(self):
            # qpos for hand
            # xpos for obj
            # xpos for target
            
            qp = self.data.qpos.ravel()
            #last 6 objects are now for target
            obj_vel = self.data.qvel[-12:-6].ravel()
            obj_pos = self.data.body_xpos[self.obj_bid].ravel()
            desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
            obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])/self.obj_length
            obj_quat = qp[19:23]
            tgt_quat = qp[-4:]
            # return np.concatenate([qp,obj_pos-desired_pos, obj_vel])
            return np.concatenate([qp[:-14], obj_pos, obj_vel, obj_orien, self.desired_orien,
                                obj_pos-desired_pos, obj_orien-self.desired_orien])


    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def reset_model(self):
        self.offset1 = (np.random.random() - 0.5) * self.random_start 
        self.offset2 = (np.random.random() - 0.5) * self.random_start
        #back and forth
        self.init_qpos[17] = self.original_qpos[17] + self.offset1
        #across palm
        self.init_qpos[16] = self.original_qpos[16] +self.offset2
        self.init_qpos[18] = .21

        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        
        self.set_state(qp, qv)

        self.sim.forward()
        self.goal = self._sample_goal().copy()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_start_pos(self, state_dict):
        self.init_qpos = state_dict['qpos']
        self.init_qvel = state_dict['qvel']

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        if(qp.shape==(22,) and qv.shape==(22,)):
            quat = self.get_qp_free_joint(qp)
            qp[16] += self.init_qpos[16]
            qp[17] += self.init_qpos[17]
            qp[18] += self.init_qpos[18]
            qp  = np.append(qp[:-3], quat)
            qp = np.append(qp, np.zeros(7))
            # qp = np.append(qp[:-6], np.zeros(14))
            
            qv  = np.append(qv, np.zeros(6))
        
        self.set_state(qp, qv)
        # self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()
    
    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if goal reached for 5 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 5:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage

    #openai/gym ManipulateEnv methods
    # ----------------------------
    def _goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])

        delta_pos = goal_a[..., :3] - goal_b[..., :3]
        d_pos = np.linalg.norm(delta_pos, axis=-1)
        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

        if self.ignore_z_target_rotation:
            # Special case: We want to ignore the Z component of the rotation.
            # This code here assumes Euler angles with xyz convention. We first transform
            # to euler, then set the Z component to be equal between the two, and finally
            # transform back into quaternions.
            euler_a = rotations.quat2euler(quat_a)
            euler_b = rotations.quat2euler(quat_b)
            euler_a[2] = euler_b[2]
            quat_a = rotations.euler2quat(euler_a)

        # Subtract quaternions and extract angle between them.
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
        d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    def set_goal_quat(self, new_quat):
        if(type(new_quat) is list):
            new_quat = np.array(new_quat)
        self.default_quat = new_quat.copy()
        self.quat_candidates.append(new_quat.copy()) 
        # self.goal = self._sample_goal().copy()

    def set_random_start(self, random_start):
        self.random_start = random_start
        self.reset_model()

    def set_mix_goals(self, mthresh):
        self.mix_goals = mthresh

    def get_goal(self):
        return self.goal

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal).astype(np.float32)
            return (success - 1.)
        else:
            d_pos, d_rot = self._goal_distance(achieved_goal, goal)
            # We weigh the difference in position to avoid that `d_pos` (in meters) is completely
            # dominated by `d_rot` (in radians).
            return -(10. * d_pos + d_rot)

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot
        return achieved_both

    def gradual_increase_difficulty(self):
        a = self.angle_from_quaternion(self.default_quat)
        new_angle = a[0] + 2
        new_quat = self.quat_from_angle_and_axis(new_angle, a[1])
        self.max_goal = new_quat
        self.set_goal_quat(new_quat)
        self.goal = self._sample_goal().copy()

    def _sample_goal(self):
        # Select a goal for the object position.
        target_pos = None
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            print('offset: ' + str(offset))
            offset[1] = 0
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3] + offset
        elif self.target_position in ['ignore', 'fixed']:
            target_pos = self.sim.data.get_joint_qpos('object:joint')[:3]
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos is not None
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation.
        target_quat = None
        if self.target_rotation == 'z':
            target_quat = self.default_quat
            if self.reward_type == 'sparse':
                target_quat = np.array(self.possible_quat[np.random.randint(0,len(self.possible_quat))])
            if(len(self.quat_candidates)>1 and np.random.random() <= self.mix_goals):
                target_quat = self.quat_candidates[np.random.randint(0,len(self.quat_candidates))]
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = self.quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = self.quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos('object:joint')
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat is not None
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)  # normalized quaternion
        goal = np.concatenate([target_pos, target_quat])
        return goal

    def mj_render(self):
        self._render_callback()
        super(BlockEnvV3, self).mj_render()

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        assert goal.shape == (7,)
        if self.target_position == 'ignore':
            # Move the object to the side since we do not care about it's position.
            goal[0] += 0.15
        self.sim.data.set_joint_qpos('target:joint', goal)
        self.sim.data.set_joint_qvel('target:joint', np.zeros(6))
        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

class DenseBlockEnvV3(BlockEnvV3):
    def __init__(self, reward_type='dense'):
        super(DenseBlockEnvV3,self).__init__(reward_type='dense')

class RandomStartEnvV3(BlockEnvV3):
    def __init__(self):
        super(RandomStartEnvV3,self).__init__(reward_type='dense')
        self.random_start = 0.1
        
    def set_random_start(self, random_start):
         super(RandomStartEnvV3,self).set_random_start(random_start)
