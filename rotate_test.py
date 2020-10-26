import pybullet as p
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PI = np.pi
TWO_PI = 2 * np.pi

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
p.changeDynamics(planeId, -1, contactStiffness=1500, contactDamping=30)
cubeStartPos = [0, 0, 0.24]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
leg_angle_arr = np.arange(0,31,2)
result = []
initial_steps = 480
sim_steps = 14400

mode = p.POSITION_CONTROL
maxforce = [3, 3, 3, 3]

w = 1.5 * PI

for leg_angle in leg_angle_arr:
    p.resetSimulation()
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1, contactStiffness=1500, contactDamping=30)
    boxId = p.loadURDF(f"same_leg_urdf/{leg_angle}_degree.urdf", cubeStartPos, cubeStartOrientation)
    print(leg_angle)
    T_pos = [0 for _ in range(4)]
    T_real = [0 for _ in range(4)]
    num = 0
    velocity = [0 for _ in range(5)]
    distance = 0
    EulerStart_tmp = 0

    p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                targetPositions=[0, -PI, -PI, 0])  # turn

    for i in range(480):
        p.stepSimulation()
        #time.sleep(1. / 240.)
    PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
    EulerStart = p.getEulerFromQuaternion(OrnStart)

    for i in range(43200):
        p.stepSimulation()
        # turn
        T_pos[0] = w / 240 * i
        T_pos[1] = w / 240 * i - math.pi
        T_pos[2] = w / 240 * i - math.pi
        T_pos[3] = w / 240 * i
        p.setJointMotorControlArray(boxId, [0, 1, 2, 3], controlMode=mode,
                                    targetPositions=[T_pos[0], T_pos[1], T_pos[2], T_pos[3]], forces=maxforce)
        #time.sleep(1. / 480.)
        if i == 7199: # start calculate after 30 seconds
            PosStart, OrnStart = p.getBasePositionAndOrientation(boxId)
            EulerStart = p.getEulerFromQuaternion(OrnStart)
            EulerStart_tmp = EulerStart[2]
        if i >= 7200:
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            EulerTest = p.getEulerFromQuaternion(OrnTest)
            if EulerTest[2] * EulerStart_tmp < 0:
                if EulerStart_tmp <= 0:
                    distance += PI - abs(EulerStart_tmp) + 0.001
                    EulerStart_tmp = PI
                else:
                    distance += EulerStart_tmp
                    EulerStart_tmp = -0.001

        if (((i + 1) % 7200 == 0) and i > 7200):
            PosTest, OrnTest = p.getBasePositionAndOrientation(boxId)
            EulerTest = p.getEulerFromQuaternion(OrnTest)
            if EulerStart_tmp < 0:
                velocity[num] = (abs(EulerTest[2] - EulerStart_tmp) + distance + 0.001) / 30
            else:
                velocity[num] = (abs(EulerTest[2] - EulerStart_tmp) + distance) / 30
            EulerStart_tmp = EulerTest[2]
            distance = 0
            num += 1

    for i in range(5):
        print(velocity[i])
    result.extend([(leg_angle, v) for v in velocity])

df = pd.DataFrame(result, columns=["leg_angle [deg]", "velocity [rads/s]"])
df.to_csv("vel_rotate.csv", index=False)