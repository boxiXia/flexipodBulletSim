import pybullet as p
import pybullet_data
import numpy as np
import time
import os
PI = np.pi


class FlexipodBulletEnv(object):

    def __init__(self, gui=True):
        robot_path = os.path.join(os.path.dirname(__file__),"urdf/robot.urdf")
        if not p.isConnected():
            self.physicsClient = p.connect(p.GUI if gui else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        p.loadURDF("plane.urdf")
        robot_start_pos = [0, 0, 0.6]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, PI / 2])
        self.robot_id = p.loadURDF(robot_path, robot_start_pos, self.robot_start_orientation,
                                   flags=p.URDF_USE_SELF_COLLISION)
        collision_index = [-1, 7, 15, 23, 31]
        for i in range(32):
            if i in collision_index:
                continue
            else:
                p.setCollisionFilterGroupMask(self.robot_id, i, 0, 0)

        self.initial_state = p.saveState()
        self.joint_pos = [0 for _ in range(32)]
        self.control_index = [i for i in range(32)]
        self.mode = p.POSITION_CONTROL
        self.max_force = [11 for _ in range(32)]
        self.constraint_id = 0

        # self.action_bound = 11 * PI  # max velocity
        # self.action_delta = PI / 240
        # self.action_range = [-self.action_bound + self.action_delta * i for i in range(action_size)]

    def reset(self):
        p.restoreState(self.initial_state)
        self.constraint_id = p.createConstraint(self.robot_id, -1, -1, -1, p.JOINT_PRISMATIC, jointAxis=[0, 0, 1],
                                                parentFramePosition=[0, 0, 0],
                                                childFramePosition=[0, 0, 0.6],
                                                childFrameOrientation=self.robot_start_orientation)
        self.prepare_robot()
        observation, _ = self.step_observation()
        return observation

    def prepare_robot(self):
        for i in range(32):
            self.joint_pos[i] = 0
        self.joint_pos[2] = -PI / 2
        self.joint_pos[10] = PI / 2
        self.joint_pos[18] = PI / 2
        self.joint_pos[26] = -PI / 2
        p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                    targetPositions=self.joint_pos)
        for i in range(240):
            p.stepSimulation()
            # time.sleep(1. / 240.)
        p.removeConstraint(self.constraint_id)

    def step(self, action):
        self.step_action(action)
        obs, cheat = self.step_observation()
        reward = self.step_reward(obs, cheat)
        return obs, reward, cheat, None

    def step_action(self, action):
        # for i in range(3):
        #     self.joint_pos[2 * (i + 1)] += action[i] / 240
        #     self.joint_pos[2 * (i + 1) + 8] -= action[i] / 240
        #     self.joint_pos[2 * (i + 1) + 16] += action[i + 3] / 240
        #     self.joint_pos[2 * (i + 1) + 24] -= action[i + 3] / 240
        for i in range(3):
            self.joint_pos[2 * (i + 1)] += action[i] / 240
            self.joint_pos[2 * (i + 1) + 8] += action[i + 3] / 240
            self.joint_pos[2 * (i + 1) + 16] += action[i + 6] / 240
            self.joint_pos[2 * (i + 1) + 24] += action[i + 9] / 240
        p.setJointMotorControlArray(self.robot_id, self.control_index, controlMode=self.mode,
                                    targetPositions=self.joint_pos,
                                    forces=self.max_force)
        p.stepSimulation()
        # time.sleep(1. / 240.)

    def step_observation(self):
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
            # obs[i + 9] = pos_test[i]
        obs[14] = PI / 2 - obs[14]
        # observation = tf.tensor(obs, dtype=tf.float)
        # observation = tf.expand_dims(observation, 0)
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


