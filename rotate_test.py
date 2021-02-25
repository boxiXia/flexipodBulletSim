import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

cubeStartPos = [0, 0, 0.24]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
leg_angle_arr = np.arange(0,31,2)
result = []
initial_steps = 480

dt = 1/240
sample_dt = 30. # interval between each measurement in seconds

sim_steps = int(sample_dt/dt*10)

sample_step = sample_dt/dt # in steps
num_samples = int(sim_steps/sample_step-1)

mode = p.POSITION_CONTROL
maxforce = [3, 3, 3, 3]

w = 2 * np.pi

for leg_angle in leg_angle_arr:
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")
    # p.changeDynamics(planeId, -1, contactStiffness=2000, contactDamping=10)
    boxId = p.loadURDF(f"final_urdf/{leg_angle}_degree.urdf", cubeStartPos, cubeStartOrientation)
    print(leg_angle)
    T_pos = np.zeros(4)
    T_real = np.zeros(4)
    num = 0
    velocity = np.zeros(num_samples)
    distance = 0
    EulerStart_tmp = 0

    p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                targetPositions=[0, -np.pi, -np.pi, 0])  # turn

    for i in range(int(1/dt)):
        p.stepSimulation()
        # time.sleep(1./480.)
    PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
    EulerStart = p.getEulerFromQuaternion(OrnStart)

    for i in range(sim_steps):
        p.stepSimulation()
        t = dt*i
        # turn
        T_pos[0] = w*t
        T_pos[1] = w*t - math.pi
        T_pos[2] = w*t - math.pi
        T_pos[3] = w*t
        p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                    targetPositions=[T_pos[0], T_pos[1], T_pos[2], T_pos[3]], forces=maxforce)
        # time.sleep(1. / 960.)
        if i == sample_step-1: # start calculate after sample_step
            PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
            EulerStart = p.getEulerFromQuaternion(OrnStart)
            EulerStart_tmp = EulerStart[2]
        if i >= sample_step:
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            EulerTest = p.getEulerFromQuaternion(OrnTest)
            if EulerTest[2] * EulerStart_tmp < 0:
                if EulerStart_tmp <= 0:
                    distance += np.pi - abs(EulerStart_tmp) + 0.00001
                    EulerStart_tmp = np.pi
                else:
                    distance += EulerStart_tmp
                    EulerStart_tmp = -0.00001

        if (((i + 1) % sample_step == 0) and i > sample_step):
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            EulerTest = p.getEulerFromQuaternion(OrnTest)
            if EulerStart_tmp < 0:
                velocity[num] = (abs(EulerTest[2] - EulerStart_tmp) + distance + 0.00001) / sample_dt
            else:
                velocity[num] = (abs(EulerTest[2] - EulerStart_tmp) + distance) / sample_dt
            EulerStart_tmp = EulerTest[2]
            distance = 0
            num += 1

    for i in range(num_samples):
        print(velocity[i])
    result.extend([(leg_angle, v) for v in velocity])

df = pd.DataFrame(result, columns=["Leg angle [deg]", "Angular velocity [rad/s]"])
df.to_csv("vel_rotate_1speed.csv", index=False)