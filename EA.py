import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time
import pybullet_data
import math
import random

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

def Simulation(w, boxId, mode, parameter):
    # t_offset, front_s_ratio, front_contact_start, front_contact_end, back_s_ratio, back_contact_start, back_contact_end
    front_stance_mid = ((parameter[2] + parameter[3]) / 2) / 180 * PI
    front_contact_angle = (parameter[3] - parameter[2]) / 180 * PI
    back_stance_mid = ((parameter[5] + parameter[6]) / 2) / 180 * PI
    back_contact_angle = (parameter[6] - parameter[5]) / 180 * PI

    gaits = [ # trotting gait #2 alternate gait
        WalkingTrot(0,p_stance_mid = front_stance_mid,contact_angle = front_contact_angle),# front left
        WalkingTrot(0,p_stance_mid = back_stance_mid,contact_angle = back_contact_angle), # back left
        WalkingTrot(parameter[0],p_stance_mid = back_stance_mid,contact_angle = back_contact_angle), # back right
        WalkingTrot(parameter[0],p_stance_mid = front_stance_mid,contact_angle = front_contact_angle)] # front right

    T_pos = [0 for _ in range(4)]
    T_test = [0 for _ in range(4)]
    counter = [0 for _ in range(4)]
    #pos_tmp = [[0 for i in range(2)] for j in range(6)]
    num = 0
    # velocity = [0 for _ in range(5)]
    # velocity_ave = 0
    velocity = 0
    cheating = False

    T_test[0] = gaits[0].GetPos(0, w, parameter[1])
    T_test[1] = -gaits[3].GetPos(0, w, parameter[1])
    T_test[2] = -gaits[2].GetPos(0, w, parameter[4])
    T_test[3] = gaits[1].GetPos(0, w, parameter[4])

    p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                targetPositions=[T_test[0], T_test[1], T_test[2], T_test[3]])


    for i in range(240):
        p.stepSimulation()
        # time.sleep(1. / 240.)
    PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
    # EulerStart = p.getEulerFromQuaternion(OrnStart)

    for i in range(2400):
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
        PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
        # if(i % 480 == 0):
        #     pos_tmp[num][0] = PosTest[0]
        #     pos_tmp[num][1] = PosTest[1]
        #     num += 1
        EulerTest = p.getEulerFromQuaternion(OrnTest)
        if(EulerTest[1] < - PI / 3):
            cheating = True
            break
        # time.sleep(1. / 240.)
    PosEnd, OrnEnd = p.getBasePositionAndOrientation(boxId)
    if cheating:
        return 0
    else:
        # for i in range(5):
        #     velocity[i] = np.sqrt(pow((pos_tmp[i][0] - pos_tmp[i + 1][0]),2) + pow((pos_tmp[i][1] - pos_tmp[i + 1][1]),2)) / 10
        #     velocity_ave += velocity[i]
        velocity = np.sqrt(pow(PosStart[0] - PosEnd[0], 2) + pow(PosStart[1] - PosEnd[1], 2)) / 10
        return velocity

def Rank(velocity, parameter):
    n = len(velocity) - 1
    for i in range(n):
        swapped = False
        for j in range(n - i):
            if velocity[j] < velocity[j + 1]:
                velocity[j], velocity[j + 1] = velocity[j + 1], velocity[j]
                for k in range(7):
                    parameter[j][k], parameter[j+1][k] = parameter[j + 1][k], parameter[j][k]
                swapped = True
        if swapped:
            swapped = False
            j = n - 1 - i
            while(j > 1):
                if velocity[j] > velocity[j - 1]:
                    velocity[j], velocity[j - 1] = velocity[j - 1], velocity[j]
                    for k in range(7):
                        parameter[j][k], parameter[j - 1][k] = parameter[j - 1][k], parameter[j][k]
                    swapped = True
                j = j - 1
            if not swapped:
                break
    return velocity, parameter

def New_individual(parameter):
    # t_offset, front_s_ratio, front_contact_start, front_contact_end, back_s_ratio, back_contact_start, back_contact_end
    parameter[0]= random.randrange(0, 100, 1) / 100
    parameter[1] = random.randrange(10, 90, 1) / 100
    parameter[2] = random.randrange(0, 180,1)
    parameter[3] = random.randrange(0, 180,1)
    parameter[4] = random.randrange(10, 90, 1) / 100
    parameter[5] = random.randrange(0, 180,1)
    parameter[6] = random.randrange(0, 180,1)
    if parameter[2] > parameter[3]:
        parameter[2], parameter[3] = parameter[3], parameter[2]
    elif parameter[2] == parameter[3] and parameter[3] < 179:
        parameter[3] += 1
    elif parameter[2] != 0:
        parameter[2] -= 1
    if parameter[5] > parameter[6]:
        parameter[5], parameter[6] = parameter[6], parameter[5]
    elif parameter[5] == parameter[6] and parameter[6] < 179:
        parameter[6] += 1
    elif parameter[5] != 0:
        parameter[5] -= 1
    return parameter

def Mutation(parameter):
    # t_offset, front_s_ratio, front_contact_start, front_contact_end, back_s_ratio, back_contact_start, back_contact_end
    rand_num = random.choice([0, 1, 2, 3, 4, 5, 6])
    if rand_num == 0:
        change = random.choice([-0.01, 0.01])
        if parameter[0] + change > 1 or parameter[0] + change < 0:
            parameter[0] -= change
        else:
            parameter[0] += change
    elif rand_num == 1:
        change = random.choice([-0.01, 0.01])
        if parameter[1] + change > 0.9 or parameter[1] + change < 0.1:
            parameter[1] -= change
        else:
            parameter[1] += change
    elif rand_num == 2:
        change = random.choice([-0.01, 0.01])
        if parameter[4] + change > 0.9 or parameter[4] + change < 0.1:
            parameter[4] -= change
        else:
            parameter[4] += change
    elif rand_num == 3:
        change = random.choice([-1,1])
        if parameter[2] + change >= parameter[3] or parameter[2] + change < 0:
            if parameter[2] - change < parameter[3]:
                parameter[2] -= change
        else:
            parameter[2] += change
    elif rand_num == 4:
        change = random.choice([-1, 1])
        if parameter[3] + change > 180 or parameter[3] + change <= parameter[2]:
            if parameter[3] - change > parameter[2]:
                parameter[3] -= change
        else:
            parameter[3] += change
    elif rand_num == 5:
        change = random.choice([-1,1])
        if parameter[5] + change >= parameter[6] or parameter[5] + change < 0:
            if parameter[5] - change < parameter[6]:
                parameter[5] -= change
        else:
            parameter[5] += change
    elif rand_num == 6:
        change = random.choice([-1, 1])
        if parameter[6] + change > 180 or parameter[6] + change <= parameter[5]:
            if parameter[6] - change > parameter[5]:
                parameter[6] -= change
        else:
            parameter[6] += change

def Swap_parameter(parameter1, parameter2):
    # t_offset, front_s_ratio, front_contact_start, front_contact_end, back_s_ratio, back_contact_start, back_contact_end
    swap_num = random.choice([1,2,3,4,5])
    if swap_num == 1:
        swap_type = random.choice([1,2,3])
        if swap_type == 1:
            parameter1[0], parameter2[0] = parameter2[0], parameter1[0]
        elif swap_type == 2:
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
        elif swap_type == 3:
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
    elif swap_num == 2:
        swap_type = random.choice([1, 2, 3, 4, 5])
        if swap_type == 1:
            parameter1[0], parameter2[0] = parameter2[0], parameter1[0]
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
        elif swap_type == 2:
            parameter1[0], parameter2[0] = parameter2[0], parameter1[0]
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
        elif swap_type == 3:
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
        elif swap_type == 4:
            parameter1[2], parameter2[2] = parameter2[2], parameter1[2]
            parameter1[3], parameter2[3] = parameter2[3], parameter1[3]
        elif swap_type == 5:
            parameter1[5], parameter2[5] = parameter2[5], parameter1[5]
            parameter1[6], parameter2[6] = parameter2[6], parameter1[6]
    elif swap_num == 3:
        swap_type = random.choice([1,2,3,4,5])
        if swap_type == 1:
            parameter1[0], parameter2[0] = parameter2[0], parameter1[0]
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
        elif swap_type == 2:
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
            parameter1[2], parameter2[2] = parameter2[2], parameter1[2]
            parameter1[3], parameter2[3] = parameter2[3], parameter1[3]
        elif swap_type == 3:
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
            parameter1[5], parameter2[5] = parameter2[5], parameter1[5]
            parameter1[6], parameter2[6] = parameter2[6], parameter1[6]
        elif swap_type == 4:
            parameter1[1], parameter2[1] = parameter2[1], parameter1[1]
            parameter1[5], parameter2[5] = parameter2[5], parameter1[5]
            parameter1[6], parameter2[6] = parameter2[6], parameter1[6]
        elif swap_type == 5:
            parameter1[4], parameter2[4] = parameter2[4], parameter1[4]
            parameter1[2], parameter2[2] = parameter2[2], parameter1[2]
            parameter1[3], parameter2[3] = parameter2[3], parameter1[3]
    elif swap_num == 4:
        parameter1[2], parameter2[2] = parameter2[2], parameter1[2]
        parameter1[3], parameter2[3] = parameter2[3], parameter1[3]
        parameter1[5], parameter2[5] = parameter2[5], parameter1[5]
        parameter1[6], parameter2[6] = parameter2[6], parameter1[6]
    return parameter1, parameter2

def Reproduce(parameter):
    survive = [[0 for i in range(7)] for j in range(100)]
    next_generation = [[0 for i in range(7)] for j in range(100)]
    for i in range(14):
        for j in range(7):
            survive[i][j] = parameter[i][j]
            survive[i+14][j] = parameter[i+20][j]
            survive[i+28][j] = parameter[i+40][j]
            survive[i+42][j] = parameter[i+60][j]
            survive[i+56][j] = parameter[i+80][j]
    for i in range(30):
        survive[i+70] = New_individual(parameter[i])
    rand_index = [i for i in range(100)]
    random.shuffle(rand_index)
    for i in range(48):
        next_generation[2*i], next_generation[2*i+1] = Swap_parameter(survive[rand_index[2*i]], survive[rand_index[2*i+1]])
    for i in range(4):
        next_generation[i + 96] = survive[i]
    for i in range(96):
        mutation_rate = random.randrange(0, 10, 1)
        if mutation_rate == 0:
            next_generation[i] = Mutation(next_generation[i])
    return parameter

physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,0.24]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("./URDF/10_degree.urdf",cubeStartPos, cubeStartOrientation)
mode = p.POSITION_CONTROL
maxforce = [5, 5, 5, 5]

w = 2 * PI
best_velocity = 0
best_parameters = [0 for _ in range(7)]
individual = [[0 for i in range(7)] for j in range(100)]
velocity = [0 for _ in range(100)]
for i in range(100):
    individual[i] = New_individual(individual[i])
    velocity[i] = Simulation(w, boxId, mode, individual[i])
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    boxId = p.loadURDF("./URDF/10_degree.urdf", cubeStartPos, cubeStartOrientation)
velocity, individual = Rank(velocity, individual)


for g in range(200):
    individual = Reproduce(individual)
    for i in range(100):
        velocity[i] = Simulation(w, boxId, mode, individual[i])
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        boxId = p.loadURDF("./URDF/10_degree.urdf", cubeStartPos, cubeStartOrientation)
    if velocity[0] > best_velocity:
        best_velocity = velocity[0]
        for i in range(7):
            best_parameters[i] = individual[0][i]

    print(best_velocity, g+1)


f = open("10_EA.txt", 'a')
f.write(str(best_velocity) + '\t')
for i in range(7):
    f.write(str(best_parameters[i]) + '\t')
f.write('\n')
f.close()
