import pybullet as p
import pybullet_data
import numpy as np
from collections import defaultdict
import time
import os
import math
from .spaces import *

PI = np.pi


class FlexipodBulletCameraEnv(object):

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

    def get_camera_img(self, reslution, field_of_view):
        self.aspect = 1
        self.near = 0.02
        self.far = 1
        self.front_camera_pos_bias = 0.01505
        self.back_camera_pos_bias = 0.1982


        projection_matrix = p.computeProjectionMatrixFOV(field_of_view, self.aspect, self.near, self.far)
        robot_info = p.getBasePositionAndOrientation(self.robot_id)
        body_pos = np.asarray(robot_info[0])
        body_ori_matrix = np.asarray(p.getMatrixFromQuaternion(robot_info[1]))
        body_x_ori = np.asarray([body_ori_matrix[1], body_ori_matrix[4], body_ori_matrix[7]])
    
        view_matrix_front = p.computeViewMatrix(body_pos - self.front_camera_pos_bias * body_x_ori, body_pos -  body_x_ori, [0, 0, 1])
        images_front = p.getCameraImage(reslution,
                            reslution,
                            view_matrix_front,
                            projection_matrix,
                            shadow=True,
                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        body_z_ori = np.asarray([body_ori_matrix[2], body_ori_matrix[5], body_ori_matrix[8]])

        view_matrix_back = p.computeViewMatrix(body_pos - self.back_camera_pos_bias * body_z_ori, body_pos - body_z_ori, [1, 0, 0])
        images_back = p.getCameraImage(reslution,
                              reslution,
                              view_matrix_back,
                              projection_matrix,
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        front_camera_img_rgb = images_front[2]
        front_camera_img_depth = images_front[3]
        back_camera_img_rgb = images_back[2]
        back_camera_img_depth = images_back[3]
        return front_camera_img_rgb, front_camera_img_depth, back_camera_img_rgb, back_camera_img_depth


    def close(self):
        p.disconnect(self.physicsClient)
