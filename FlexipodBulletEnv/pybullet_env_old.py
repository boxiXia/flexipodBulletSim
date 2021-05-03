import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from .spaces import *
PI = np.pi


class FlexipodBulletEnv(object):

    def __init__(self, gui=True, control_mode=True):
        robot_path = os.path.join(os.path.dirname(__file__),"urdf_old/robot.urdf")
        if not p.isConnected():
            self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        self.mode = p.POSITION_CONTROL if control_mode else p.VELOCITY_CONTROL

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        robot_start_pos = [0, 0, 0.6]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, PI / 2])  # make robot vertical
        self.robot_id = p.loadURDF(robot_path, robot_start_pos, self.robot_start_orientation,
                                   flags=p.URDF_USE_SELF_COLLISION)  # load urdf
        collision_index = [-1, 7, 15, 23, 31]  # only consider self collision for main body and four lower legs
        for i in range(32):
            if i in collision_index:
                continue
            else:
                p.setCollisionFilterGroupMask(self.robot_id, i, 0, 0)

        self.joint_pos = [0 for _ in range(32)]  # robot have 32 link, only 12 are not fixed
        self.joint_vel = [0 for _ in range(32)]
        self.control_index = [i for i in range(32)]
        self.max_force = [11 for _ in range(32)]
        self.constraint_id = 0
        self.initial_state = p.saveState()

        self.observation_space = Box(low=-PI, high=PI, shape=(15, 2))
        self.action_space = Box(low=-PI, high=PI, shape=(12,)) if control_mode else Box(low=-8, high=8, shape=(12,))

    def reset(self):
        """
        reset the robot to initial state so it can be trained again
        add constraint so robot can start to train in a stand pose
        :return:
        observation (object): agent's observation of the current environment
        """
        p.restoreState(self.initial_state)
        self.constraint_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_PRISMATIC, jointAxis=[0, 0, 1],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0.6],
                                                childFrameOrientation=self.robot_start_orientation)
        self.prepare_robot()
        observation, _ = self.step_observation()
        return observation

    def prepare_robot(self):
        """
        change the initial angle of the legs to help robot stand
        after 1 sec, the robot stand on the ground then remove the constraint and start to train
        :return:
        """
        for i in range(32):
            self.joint_pos[i] = 0
            self.joint_vel[i] = 0
        self.joint_pos[2] = -PI / 2
        self.joint_pos[10] = PI / 2
        self.joint_pos[18] = PI / 2
        self.joint_pos[26] = -PI / 2
        p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.joint_pos)
        for i in range(240):
            p.stepSimulation()
            # time.sleep(1. / 240.)
        p.removeConstraint(self.constraint_id)

    def step(self, action=None):
        """
        step the action and get observation, reward and cheating info
        :param action: the motor's velocity/ angle to take in this time step
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the robot cheat
        """
        if action is not None:
            self.step_action(action)
        obs, cheat = self.step_observation()
        reward = self.step_reward(obs, cheat)
        return obs, reward, cheat, None

    def step_action(self, action):
        """
        taking the action for one time step
        :param action:
        :return:
        """
        if self.mode == p.POSITION_CONTROL:
            for i in range(3):
                self.joint_pos[2 * (i + 1)] += action[i] / 240
                self.joint_pos[2 * (i + 1) + 8] += action[i + 3] / 240
                self.joint_pos[2 * (i + 1) + 16] += action[i + 6] / 240
                self.joint_pos[2 * (i + 1) + 24] += action[i + 9] / 240
            # control 12 motors out of 32 links
            p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                        targetPositions=self.joint_pos,
                                        forces=self.max_force)
        else:
            for i in range(3):
                self.joint_vel[2 * (i + 1)] = action[i]
                self.joint_vel[2 * (i + 1) + 8] = action[i + 3]
                self.joint_vel[2 * (i + 1) + 16] = action[i + 6]
                self.joint_vel[2 * (i + 1) + 24] = action[i + 9]
            p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                        targetVelocities=self.joint_vel,
                                        forces=self.max_force)
        p.stepSimulation()
        # time.sleep(1. / 240.)

    def step_observation(self):
        """
        getting observation after taking the actions, and determine if the robot cheat
        :return:
        """
        obs = np.zeros(15)
        cheating = False
        pos_test, orn_test = p.getBasePositionAndOrientation(self.robot_id)
        euler_test = p.getEulerFromQuaternion(orn_test)
        for i in range(3):
            test1, _, _, _ = p.getJointState(bodyUniqueId=self.robot_id, jointIndex=2 * (i + 1))
            test2, _, _, _ = p.getJointState(bodyUniqueId=self.robot_id, jointIndex=2 * (i + 1) + 8)
            test3, _, _, _ = p.getJointState(bodyUniqueId=self.robot_id, jointIndex=2 * (i + 1) + 16)
            test4, _, _, _ = p.getJointState(bodyUniqueId=self.robot_id, jointIndex=2 * (i + 1) + 24)
            obs[i] = test1
            obs[i + 3] = test2
            obs[i + 6] = test3
            obs[i + 9] = test4
            obs[i + 12] = euler_test[i]
            obs[i + 9] = pos_test[i]

        for i in range(15):
            if obs[i] > PI:
                obs[i] = -PI + obs[i] - PI
            elif obs[i] < -PI:
                obs[i] = PI - obs[i] + PI

        obs[14] = PI / 2 - obs[14]
        pos_31 = p.getLinkState(self.robot_id, 31)
        pos_29 = p.getLinkState(self.robot_id, 29)
        pos_23 = p.getLinkState(self.robot_id, 23)
        pos_21 = p.getLinkState(self.robot_id, 21)
        if pos_test[2] < 0.2 or pos_31[0][2] < 0.06 or pos_29[0][2] < 0.05 or abs(obs[12]) > PI / 4 \
                or pos_31[0][2] > pos_29[0][2] or pos_23[0][2] > pos_21[0][2] \
                or pos_31[0][2] > 0.16 or pos_23[0][2] > 0.16:
            cheating = True
        return obs, cheating

    def step_reward(self, obs, cheat):
        """
        computing the reward
        :param obs:
        :param cheat:
        :return:
        """
        pos_31 = p.getLinkState(self.robot_id, 31)
        pos_23 = p.getLinkState(self.robot_id, 23)
        if cheat:
            reward = -500
        elif pos_23[0][2] > 0.135 and pos_31[0][2] > 0.135:
            reward = -10
        else:
            if -self.joint_pos[26] + self.joint_pos[30] > PI / 2:
                leg_reward1 = PI - (-self.joint_pos[26] + self.joint_pos[30])
            else:
                leg_reward1 = -self.joint_pos[26] + self.joint_pos[30]
            if self.joint_pos[18] - self.joint_pos[22] > PI / 2:
                leg_reward2 = PI - (self.joint_pos[18] - self.joint_pos[22])
            else:
                leg_reward2 = self.joint_pos[18] - self.joint_pos[22]
            leg_pos_reward = 0.2 - abs(pos_23[0][1]) - abs(pos_31[0][1])
            pos_test, _ = p.getBasePositionAndOrientation(self.robot_id)
            reward = PI / 2 - abs(obs[12]) + leg_reward1 + leg_reward2 + 10 * (leg_pos_reward - abs(pos_test[1])) \
                     + pos_test[2]

        return reward

    def close(self):
        p.disconnect(self.physicsClient)
