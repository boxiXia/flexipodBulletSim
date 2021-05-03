import pybullet as p
import pybullet_data
import numpy as np
from collections import defaultdict
import time
import os
import math
from .spaces import *

PI = np.pi


class FlexipodBulletEnv(object):

    def __init__(self, gui=True, control_mode=True):
        robot_path = os.path.join(os.path.dirname(__file__), "urdf/robot.urdf")
        if not p.isConnected():
            self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)
        self.mode = p.POSITION_CONTROL if control_mode else p.VELOCITY_CONTROL

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        robot_start_pos = [0, 0, 0.6]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, PI / 2])  # make robot vertical
        self.robot_id = p.loadURDF(robot_path, robot_start_pos, self.robot_start_orientation)#,
        #                            flags=p.URDF_USE_SELF_COLLISION)  # load urdf
        # collision_index = [-1, 7, 15, 23, 31]  # only consider self collision for main body and four lower legs
        # for i in range(32):
        #     if i in collision_index:
        #         continue
        #     else:
        #         p.setCollisionFilterGroupMask(self.robot_id, i, 0, 0)

        self.joint_pos = [0 for _ in range(32)]  # robot have 32 link, only 12 are not fixed
        self.joint_vel = [0 for _ in range(32)]
        self.control_index = [i for i in range(32)]
        self.max_force = [11 for _ in range(32)]
        self.constraint_id = 0
        self.initial_state = p.saveState()

        max_joint_vel = 10
        # name of the returned message
        REC_NAME = np.array([
            # name,         size,      min,          max
            ("joint_pos", 12 * 2, -1., 1.),  # joint cos(angle) sin(angle) [rad]
            ("joint_vel", 12, -max_joint_vel, max_joint_vel),  # joint velocity [rad/s]
            ("actuation", 12, -1., 1.),  # joint actuation [-1,1]
            ("ang_vel", 3, -30., 30.),  # base link (body) angular velocity [rad/s]
            ("com_vel", 3, -2., 2.),  # base link (body) velocity
            ("com_pos", 3., -1., 1.),  # base link (body) position
            ("orientation", 6, -1., 1.),  # base link (body) orientation
        ], dtype=[('name', 'U14'), ('size', 'i4'), ('min', 'f4'), ('max', 'f4')])
        REC_SIZE = REC_NAME["size"]
        REC_MIN = REC_NAME["min"]
        REC_MAX = REC_NAME["max"]
        OBS_NAME = ("joint_pos", "joint_vel", "actuation", "ang_vel", "com_vel", "com_pos", "orientation",)
        self.ID = defaultdict(None, {name: k for k, (name, _, _, _) in enumerate(REC_NAME)})
        self.num_observation = 63
        self.min_observation = np.hstack([
            np.ones(REC_SIZE[self.ID[name]]) * REC_MIN[self.ID[name]] for name in OBS_NAME])  # [:-2]
        self.max_observation = np.hstack([
            np.ones(REC_SIZE[self.ID[name]]) * REC_MAX[self.ID[name]] for name in OBS_NAME])  # [:-2]

        # self.min_observation = np.tile(self.min_observation, (self.num_observation, 1)).astype(np.float32)
        # self.max_observation = np.tile(self.max_observation, (self.num_observation, 1)).astype(np.float32)

        self.observation_space = Box(
            low=self.min_observation,
            high=self.max_observation,
            dtype=np.float32)
        self.action_space = Box(low=-PI, high=PI, shape=(12,)) if control_mode else Box(low=-2, high=2, shape=(12,))
        self.max_action = 1/120 if control_mode else 2

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
        self.joint_pos[3] = PI / 2
        self.joint_pos[11] = -PI / 2
        self.joint_pos[17] = PI / 2
        self.joint_pos[25] = -PI / 2
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
            cmd_action = np.multiply(action, self.max_action).astype(np.float32)
            self.step_action(cmd_action)
        obs, cheat = self.step_observation()
        reward = self.step_reward(obs)
        return obs, reward, cheat, None

    def step_action(self, action):
        """
        taking the action for one time step
        :param action:
        :return:
        """
        if self.mode == p.POSITION_CONTROL:
            for i in range(4):
                self.joint_pos[1 + 8 * i] += action[0 + i * 3]
                self.joint_pos[3 + 8 * i] += action[1 + i * 3]
                self.joint_pos[6 + 8 * i] += action[2 + i * 3]
            # control 12 motors out of 32 links
            p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                        targetPositions=self.joint_pos,
                                        forces=self.max_force)
        else:
            for i in range(4):
                self.joint_vel[1 + 8 * i] = action[0 + i * 3]
                self.joint_vel[3 + 8 * i] = action[1 + i * 3]
                self.joint_vel[6 + 8 * i] = action[2 + i * 3]
            # control 12 motors out of 32 links
            p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                        targetVelocities=self.joint_vel,
                                        forces=self.max_force)
        p.stepSimulation()
        # time.sleep(1. / 240.)

    def step_observation(self):
        """
        getting observation after taking the actions, and determine if the robot cheat
        obs = "joint_pos","joint_vel","actuation","ang_vel","com_vel","com_pos","orientation"
                  24          12           12         3         3         3          6
        :return:
        """
        obs = np.zeros(self.num_observation)
        cheating = False
        pos_test, orn_test = p.getBasePositionAndOrientation(self.robot_id)
        euler_test = p.getEulerFromQuaternion(orn_test)
        base_vel, base_ang = p.getBaseVelocity(self.robot_id)
        for i in range(4):
            obs[0 + i * 6], obs[24 + i * 3], _, obs[36 + i * 3] = p.getJointState(bodyUniqueId=self.robot_id,
                                                                                  jointIndex=1 + 8 * i)
            obs[2 + i * 6], obs[25 + i * 3], _, obs[37 + i * 3] = p.getJointState(bodyUniqueId=self.robot_id,
                                                                                  jointIndex=3 + 8 * i)
            obs[4 + i * 6], obs[26 + i * 3], _, obs[38 + i * 3] = p.getJointState(bodyUniqueId=self.robot_id,
                                                                                  jointIndex=6 + 8 * i)
        for i in range(3):
            obs[i + 54] = pos_test[i]
            obs[i + 48] = base_ang[i]
            obs[i + 51] = base_vel[i]
        obs[57] = math.sin(euler_test[1])
        obs[58] = -math.sin(euler_test[0]) * math.cos(euler_test[1])
        obs[59] = math.cos(euler_test[0]) * math.cos(euler_test[1])
        obs[60] = math.cos(euler_test[1]) * math.cos(euler_test[2])
        obs[61] = math.sin(euler_test[0]) * math.sin(euler_test[1]) * math.cos(euler_test[2]) \
                  + math.cos(euler_test[0]) * math.sin(euler_test[2])
        obs[62] = -math.cos(euler_test[0]) * math.sin(euler_test[1]) * math.cos(euler_test[2]) \
                  + math.sin(euler_test[0]) * math.sin(euler_test[2])
        for i in range(12):
            obs[2 * i + 1] = math.sin(obs[2 * i])
            obs[2 * i] = math.cos(obs[2 * i])
            obs[i + 36] = obs[i + 36] / self.max_force[0]
        obs = (obs / self.max_observation)
        # for i in range(15):
        #     if obs[i] > PI:
        #         obs[i] = -PI + obs[i] - PI
        #     elif obs[i] < -PI:
        #         obs[i] = PI - obs[i] + PI
        orientation_z = obs[59]
        com_z = pos_test[2]
        cheating = True if (orientation_z < 0.6) or (com_z < 0.2) else False
        return obs, cheating

    def step_reward(self, obs):
        """
        computing the reward
        :param obs:
        :return:
        """
        actuation = np.zeros(12)
        orientation_z = obs[59]
        com_z = obs[56]
        uph_cost = orientation_z + com_z - 0.15
        for i in range(12):
            actuation[i] = obs[i + 36]
        quad_ctrl_cost = 0.1 * np.square(actuation).sum()  # quad control cost
        reward = uph_cost - quad_ctrl_cost
        return reward

    def close(self):
        p.disconnect(self.physicsClient)
