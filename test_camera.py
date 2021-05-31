import numpy as np
import matplotlib.pyplot as plt
# from flexipod_env import FlexipodEnv
from FlexipodBulletEnv import FlexipodBulletCameraEnv
import pybullet as p
# env = FlexipodEnv(dof = 12)
env = FlexipodBulletCameraEnv(gui=False, control_mode=True)

while (True):
    p.stepSimulation()

    front_rgb, front_depth, back_rgb, back_depth = env.get_camera_img(256, 120)
    img = np.reshape(front_rgb, (256, 256, 4)) * 1. / 255.
    plt.imshow(img)
    plt.show()
