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
                 # angle at the middle of the stance phase [rad]
                 p_stance_mid=1. * np.pi,
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
        # w_stance, (time normalized) average angular velocity of stance phase [rad]
        this.ws = this.c / (this.s)
        this.wa = (TWO_PI - this.c) / (
            1 - this.s)  # w_air, (time normalized) average angular velocity of air phase [rad]

        this.p_stance_high = p_stance_mid + this.c / 2.0
        #print(np.rad2deg(this.p_stance_low), np.rad2deg(p_stance_mid), np.rad2deg(this.p_stance_high))

        # get the initial position
        if this.ot < this.s:  # if normalized time offset < stance ratio
            this.pos = this.p_stance_low + this.c / this.s * this.ot
        else:
            this.pos = this.p_stance_low + this.c + \
                (TWO_PI - this.c) / (1 - this.s) * (this.ot - this.s)

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
        ws = this.c / \
            (s)  # w_stance, (time normalized) average angular velocity of stance phase [rad]
        # w_air, (time normalized) average angular velocity of air phase [rad]
        wa = (TWO_PI - this.c) / (1 - s)
        # update the converted (normalized) time this.tc

        this.s = s  # stance ratio = stance_time/cycle_time [unitless]
        this.ws = ws
        this.wa = wa

        if this.tc < this.s:  # 0-s
            this.pos = (this.p_stance_low + this.ws * this.tc) % TWO_PI
        else:  # s-1
            this.pos = (this.p_stance_low + this.c +
                        this.wa * (this.tc - this.s)) % TWO_PI

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
        # normalized phase difference [0-1][unitless]
        dnt = this.w * dt / TWO_PI
        # # current converted time [0-1,unitless]
        this.tc = (this.tc + dnt) % 1.0

        # todo sign
        if this.tc < this.s:  # 0-s
            this.pos = (this.p_stance_low + this.ws * this.tc) % TWO_PI
        else:  # s-1
            this.pos = (this.p_stance_low + this.c +
                        this.wa * (this.tc - this.s)) % TWO_PI
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


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
# p.changeDynamics(planeId, -1, contactStiffness=1500, contactDamping=30)
cubeStartPos = [0, 0, 0.24]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
leg_angle_arr = np.arange(0, 31, 2)
result = []
initial_steps = 480

dt = 1/240
sample_dt = 10 # interval between each measurement in seconds
sample_step = int(sample_dt/dt) # in steps

sim_steps = sample_step*10

num_samples = int(sim_steps/sample_step-1)

mode = p.POSITION_CONTROL
maxforce = [3, 3, 3, 3]

w = 2 * PI
parameter = [0 for _ in range(7)]
parameter[0] = 0.2
parameter[1] = 0.6 
parameter[2] = 25
parameter[3] = 155
parameter[4] = 0.6 
parameter[5] = 25
parameter[6] = 155 

for leg_angle in leg_angle_arr:
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    # p.changeDynamics(planeId, -1, contactStiffness=1500, contactDamping=30)
    boxId = p.loadURDF(f"final_urdf/{leg_angle}_degree.urdf", cubeStartPos, cubeStartOrientation)
    # boxId = p.loadURDF("same_leg_urdf/24_degree.urdf", cubeStartPos, cubeStartOrientation)
    # p.changeDynamics(boxId, 1, contactStiffness = 1000, contactDamping = 1000)

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
    # gaits = [  # rotate
    #     WalkingTrot(0, p_stance_mid=front_stance_mid,
    #                 contact_angle=front_contact_angle),  # front left
    #     WalkingTrot(0, p_stance_mid=back_stance_mid,
    #                 contact_angle=back_contact_angle),  # back left
    #     WalkingTrot(parameter[0], p_stance_mid=back_stance_mid,
    #                 contact_angle=back_contact_angle),  # back right
    #     WalkingTrot(parameter[0], p_stance_mid=front_stance_mid, contact_angle=front_contact_angle)]  # front right
    # jointPosition = np.zeros(1440)
    # jointVelocity = np.zeros(1440)
    # appliedJointMotorTorque = np.zeros(1440)

    # 1 left_front
    # 2 right_front
    # 3 right_back
    # 4 left_back
    T_pos = np.zeros(4)#[0 for _ in range(4)]
    T_test = np.zeros(4)#[0 for _ in range(4)]
    counter = np.zeros(4)#[0 for _ in range(4)]
    pos_tmp = np.zeros((2,6))#[[0 for i in range(2)] for j in range(6)]
    num = 0
    velocity = np.zeros(num_samples)#[0 for _ in range(num_samples)]

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
        #time.sleep(1. / 240.)
    PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
    # EulerStart = p.getEulerFromQuaternion(OrnStart)

    for i in range(sim_steps):
        p.stepSimulation()
        t = i*dt
        T_pos[0] = gaits[0].GetPos(t, w, parameter[1])
        T_pos[1] = gaits[3].GetPos(t, w, parameter[1])
        T_pos[2] = gaits[2].GetPos(t, w, parameter[4])
        T_pos[3] = gaits[1].GetPos(t, w, parameter[4])
        for a in range(4):
            if T_pos[a] < (T_test[a] - 3):
                counter[a] += 1
            T_test[a] = T_pos[a]

        T_pos[0] =  gaits[0].GetPos(t, w, parameter[1]) + TWO_PI * counter[0]
        T_pos[1] = -gaits[3].GetPos(t, w,parameter[1]) - TWO_PI * counter[1]
        T_pos[2] = -gaits[2].GetPos(t, w,parameter[4]) - TWO_PI * counter[2]
        T_pos[3] =  gaits[1].GetPos(t, w, parameter[4]) + TWO_PI * counter[3]

        p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                    targetPositions=[
                                        T_pos[0], T_pos[1], T_pos[2], T_pos[3]],
                                    forces=maxforce)
        if i == sample_step-1:  # Start calculate after 10 seconds
            PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
        elif ((i + 1) % sample_step == 0):
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            velocity[num] = np.sqrt(
                pow(PosStart[0] - PosTest[0], 2) + pow(PosStart[1] - PosTest[1], 2)) / sample_dt
            PosStart = PosTest
            num += 1
        #time.sleep(1. / 240.)
    #PosEnd, OrnEnd = p.getBasePositionAndOrientation(boxId)

    for i in range(num_samples):
        print(velocity[i])
    result.extend([(leg_angle, v) for v in velocity])
result = np.asarray(result)
# print(result)

df = pd.DataFrame(result, columns=["Leg angle [deg]", "Velocity [m/s]"])
df.to_csv("vel_bounding_test.csv", index=False)
