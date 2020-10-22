import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import math
import random
import pandas as pd

plt.style.use('seaborn-whitegrid')

PI = np.pi
TWO_PI = 2 * np.pi

class WalkingTrot():
    """
  trotting gait
  @author: HonboZhu, BoxiXia
  """

    def __init__(this,
                 ot=0,  # normalized time offset [0-1], [unitless]
                 w=0.1 * np.pi,  # average angular velocity [rad/s]
                 s=0.5,  # stance ratio = stance_time/cycle_time [unitless]
                 contact_angle=120 / 180 * PI,  # contact_angle [rad]
                 p_stance_mid=1. * np.pi,  # angle at the middle of the stance phase [rad]
                 t=0,  # time at current step
                 ):
        this.ot = ot  # normalized time offset [0-1], [unitless]
        assert (0 <= this.ot <= 1)
        this.w = w  # average angular velocity [rad/s]
        this.c = contact_angle  # contact_angle [rad]
        #         this.contact_angle
        this.p_stance_mid = p_stance_mid
        this.p_stance_low = p_stance_mid - this.c / 2.0

        this.t = t  # time at current step
        this.tc = this.ot  # converted time [0-1,unitless]

        this.T = 2 * np.pi / this.w  # cycle time

        this.s = s  # stance ratio = stance_time/cycle_time [unitless]
        this.ws = this.c / (this.s)  # w_stance, (time normalized) average angular velocity of stance phase [rad]
        this.wa = (TWO_PI - this.c) / (
                    1 - this.s)  # w_air, (time normalized) average angular velocity of air phase [rad]

        this.p_stance_high = p_stance_mid + this.c / 2.0
        #print(np.rad2deg(this.p_stance_low), np.rad2deg(p_stance_mid), np.rad2deg(this.p_stance_high))

        # get the initial position
        if this.ot < this.s:  # if normalized time offset < stance ratio
            this.pos = this.p_stance_low + this.c / this.s * this.ot
        else:
            this.pos = this.p_stance_low + this.c + (TWO_PI - this.c) / (1 - this.s) * (this.ot - this.s)

    #     @njit
    #     def UpdateStanceRatio(this,s):
    #         """
    #         update the stance ratio s (=: stance_time/cycle_time [unitless])
    #         """
    #         ws = this.c/(s) # w_stance, (time normalized) average angular velocity of stance phase [rad]
    #         wa = (TWO_PI-this.c)/(1-s) # w_air, (time normalized) average angular velocity of air phase [rad]
    #         # update the converted (normalized) time this.tc

    #         pos_raw = (this.pos -this.p_stance_low)%TWO_PI
    #         if pos_raw<0:
    #             pos_raw+=TWO_PI # convert to 0-2PI
    #         if pos_raw<this.c:
    #             this.tc = pos_raw/ws
    #         else:
    #             this.tc = (pos_raw-this.c)/wa+s

    #         this.s = s # stance ratio = stance_time/cycle_time [unitless]
    #         this.ws = ws
    #         this.wa = wa

    def UpdateStanceRatio(this, s):
        """
        update the stance ratio s (=: stance_time/cycle_time [unitless])
        """
        ws = this.c / (s)  # w_stance, (time normalized) average angular velocity of stance phase [rad]
        wa = (TWO_PI - this.c) / (1 - s)  # w_air, (time normalized) average angular velocity of air phase [rad]
        # update the converted (normalized) time this.tc

        this.s = s  # stance ratio = stance_time/cycle_time [unitless]
        this.ws = ws
        this.wa = wa

        if this.tc < this.s:  # 0-s
            this.pos = (this.p_stance_low + this.ws * this.tc) % TWO_PI
        else:  # s-1
            this.pos = (this.p_stance_low + this.c + this.wa * (this.tc - this.s)) % TWO_PI

    def GetPos(this, t, w=None, s=None):
        """
        return the position [0,2pi] [rad] given the current time [s]
        """
        if w is not None:
            this.w = w
        if (s is not None) and s != this.s:
            this.UpdateStanceRatio(s)
        dt = t - this.t  # differenct in raw time
        this.t = t
        dnt = this.w * dt / TWO_PI  # normalized phase difference [0-1][unitless]
        this.tc = (this.tc + dnt) % 1.0  # # current converted time [0-1,unitless]

        # todo sign
        if this.tc < this.s:  # 0-s
            this.pos = (this.p_stance_low + this.ws * this.tc) % TWO_PI
        else:  # s-1
            this.pos = (this.p_stance_low + this.c + this.wa * (this.tc - this.s)) % TWO_PI
        return this.pos

    def _UpdateAngularVelocity(this, t, w):
        this.w = w  # TODO direction


# gaits = [ # trotting gait #2 alternate gait
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI),# front left
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI), # back left
#     WalkingTrot(0.5,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI), # back right
#     WalkingTrot(0.5,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI)] # front right
#  offset 0.01 (stance (0-90) 0.1 contact angle (0-180) 0.1) two points  stance_ratio 0.1 - 0.9

# gaits = [ # trotting gait #2 alternate gait
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI),# front left
#     WalkingTrot(0.7,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI), # back left
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI), # back right
#     WalkingTrot(0.7,p_stance_mid = 1./2.*np.pi,contact_angle = 145/180*PI)] # front right

# gaits = [  # bounding gait
#     WalkingTrot(0, p_stance_mid=1. / 2. * np.pi, contact_angle=160 / 180 * PI),  # front left
#     WalkingTrot(0.4, p_stance_mid=1. / 2. * np.pi, contact_angle=160 / 180 * PI),  # back left
#     WalkingTrot(0.4, p_stance_mid=1. / 2. * np.pi, contact_angle=160 / 180 * PI),  # back right
#     WalkingTrot(0, p_stance_mid=1. / 2. * np.pi, contact_angle=160 / 180 * PI)  # front right
# ]

# gaits = [ # bounding gait same phase
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI),# front left
#     WalkingTrot(0.20,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI), # back left
#     WalkingTrot(0.20,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI), # back right
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI) # front right
# ]

# gaits = [ # bounding gait same phase
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI),# front left
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI), # back left
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI), # back right
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 180/180*PI) # front right
# ]


# gaits = [ # crawl gait #2
#     WalkingTrot(0,p_stance_mid = 1./2.*np.pi,contact_angle = 120/180*PI),# front left
#     WalkingTrot(0.75,p_stance_mid = 1./2.*np.pi,contact_angle = 120/180*PI), # back left
#     WalkingTrot(0.25,p_stance_mid = 1./2.*np.pi,contact_angle = 120/180*PI), # back right
#     WalkingTrot(0.5,p_stance_mid = 1./2.*np.pi,contact_angle = 120/180*PI) # front right
# ]


# t_arr = np.linspace(0,4,1000)
# w_arr = PI*np.ones_like(t_arr)
# s_arr = np.linspace(0.5,0.5,1000)
#
# angle1 = np.array([gaits[0].GetPos(t, w, s) for t, w, s in zip(t_arr, w_arr, s_arr)])
# angle2 = np.array([gaits[1].GetPos(t, w, s) for t, w, s in zip(t_arr, w_arr, s_arr)])
# angle3 = np.array([gaits[2].GetPos(t, w, s) for t, w, s in zip(t_arr, w_arr, s_arr)])
# angle4 = np.array([gaits[3].GetPos(t, w, s) for t, w, s in zip(t_arr, w_arr, s_arr)])


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
p.changeDynamics(planeId, -1, contactStiffness=1500, contactDamping=30)
cubeStartPos = [0, 0, 0.24]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
leg_angle_arr = np.arange(0,31,5)
result = []
initial_steps = 480
sim_steps = 14400
for leg_angle in leg_angle_arr:
    boxId = p.loadURDF(f"same_leg_urdf/{leg_angle}_degree.urdf", cubeStartPos, cubeStartOrientation)
    # p.changeDynamics(boxId, 1, contactStiffness = 1000, contactDamping = 1000)
    mode = p.POSITION_CONTROL
    maxforce = [5, 5, 5, 5]

    w = 3 * PI
    parameter = [0 for _ in range(7)]
    parameter[0] = 0.5  # 0.51
    parameter[1] = 0.8  # 0.74
    parameter[2] = 17.5  # 7
    parameter[3] = 162.5  # 113
    parameter[4] = 0.8  # 0.67
    parameter[5] = 17.5  # 95
    parameter[6] = 162.5  # 134
    front_stance_mid = ((parameter[2] + parameter[3]) / 2) / 180 * PI
    front_contact_angle = (parameter[3] - parameter[2]) / 180 * PI
    back_stance_mid = ((parameter[5] + parameter[6]) / 2) / 180 * PI
    back_contact_angle = (parameter[6] - parameter[5]) / 180 * PI

    gaits = [  # bounding gait
        WalkingTrot(0, p_stance_mid=front_stance_mid, contact_angle=front_contact_angle),  # front left
        WalkingTrot(parameter[0], p_stance_mid=back_stance_mid, contact_angle=back_contact_angle),  # back left
        WalkingTrot(parameter[0], p_stance_mid=back_stance_mid, contact_angle=back_contact_angle),  # back right
        WalkingTrot(0, p_stance_mid=front_stance_mid, contact_angle=front_contact_angle)]  # front right
    # gaits = [  # trotting gait #2 alternate gait
    #     WalkingTrot(0, p_stance_mid=front_stance_mid, contact_angle=front_contact_angle),  # front left
    #     WalkingTrot(0, p_stance_mid=back_stance_mid, contact_angle=back_contact_angle),  # back left
    #     WalkingTrot(parameter[0], p_stance_mid=back_stance_mid, contact_angle=back_contact_angle),  # back right
    #     WalkingTrot(parameter[0], p_stance_mid=front_stance_mid, contact_angle=front_contact_angle)]  # front right
    # jointPosition = np.zeros(1440)
    # jointVelocity = np.zeros(1440)
    # appliedJointMotorTorque = np.zeros(1440)

    # 1 left_front
    # 2 right_front
    # 3 right_back
    # 4 left_back
    T_pos = [0 for _ in range(4)]
    T_test = [0 for _ in range(4)]
    counter = [0 for _ in range(4)]
    pos_tmp = [[0 for i in range(2)] for j in range(6)]
    num = 0
    velocity = [0 for _ in range(5)]
    cheating = False

    T_test[0] = gaits[0].GetPos(0, w, parameter[1])
    T_test[1] = -gaits[3].GetPos(0, w, parameter[1])
    T_test[2] = -gaits[2].GetPos(0, w, parameter[4])
    T_test[3] = gaits[1].GetPos(0, w, parameter[4])

    p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                targetPositions=[T_test[0], T_test[1], T_test[2], T_test[3]])
    # p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
    #                             targetPositions=[PI / 4, PI / 4 , PI / 4, PI / 4])

    for i in range(initial_steps):
        p.stepSimulation()
        time.sleep(1. / 240.)
    PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
    # EulerStart = p.getEulerFromQuaternion(OrnStart)

    for i in range(sim_steps):
        p.stepSimulation()

        T_pos[0] = gaits[0].GetPos(i / 240, w, parameter[1])
        T_pos[1] = gaits[3].GetPos(i / 240, w, parameter[1])
        T_pos[2] = gaits[2].GetPos(i / 240, w, parameter[4])
        T_pos[3] = gaits[1].GetPos(i / 240, w, parameter[4])
        for a in range(4):
            if T_pos[a] < (T_test[a] - 3):
                counter[a] += 1
            T_test[a] = T_pos[a]

        T_pos[0] = gaits[0].GetPos(i / 240, w, parameter[1]) + TWO_PI * counter[0]
        T_pos[1] = -gaits[3].GetPos(i / 240, w, parameter[1]) - TWO_PI * counter[1]
        T_pos[2] = -gaits[2].GetPos(i / 240, w, parameter[4]) - TWO_PI * counter[2]
        T_pos[3] = gaits[1].GetPos(i / 240, w, parameter[4]) + TWO_PI * counter[3]

        p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                    targetPositions=[T_pos[0], T_pos[1], T_pos[2], T_pos[3]],
                                    forces=maxforce)
        if (((i + 1) % 2400 == 0) and i > 2400):
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            velocity[num] = np.sqrt(pow(PosStart[0] - PosTest[0], 2) + pow(PosStart[1] - PosTest[1], 2)) / 10
            PosStart = PosTest
            num += 1
        time.sleep(1. / 240.)
    #PosEnd, OrnEnd = p.getBasePositionAndOrientation(boxId)

    for i in range(5):
        print(velocity[i])
    result.extend([(leg_angle, v) for v in velocity])
result = np.asarray(result)
# print(result)

df = pd.DataFrame(result, columns=["leg_angle [deg]", "velocity [m/s]"])
df.to_csv("vel_alternate_gait.csv", index=False)

