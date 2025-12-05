import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import random
import time

class ArmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render=False):
        super().__init__()

        self.render = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # action = 1 value in [-1,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # observation = [theta, target_x, target_y]
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -1.0, -1.0]),
            high=np.array([ np.pi, 1.0, 1.0]),
            dtype=np.float32
        )

        self.L = 1.0
        self._make_arm()
        self.target = None
        self.marker_id = None   # 🔥 store debug line id
        self.steps = 0

    def _make_arm(self):
        L = self.L

        self.base = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0,0,0]
        )

        link_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L/2, 0.05, 0.05],
            collisionFramePosition=[L/2, 0, 0]
        )
        link_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L/2, 0.05, 0.05],
            visualFramePosition=[L/2, 0, 0],
            rgbaColor=[1,0,0,1]
        )

        self.arm = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0,0,0],

            linkMasses=[1],
            linkCollisionShapeIndices=[link_col],
            linkVisualShapeIndices=[link_vis],
            linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[0,0,1]]
        )

    def _new_target(self):
        r = random.uniform(0.3, 1.0)
        ang = random.uniform(-math.pi, math.pi)
        return [r*math.cos(ang), r*math.sin(ang)]

    def wrap(self, a):
        return (a + math.pi) % (2*math.pi) - math.pi

    def _get_state(self):
        theta = p.getJointState(self.arm, 0)[0]
        theta = self.wrap(theta)
        return np.array([theta, self.target[0], self.target[1]], dtype=np.float32)

    def _compute_reward(self, theta):
        desired = math.atan2(self.target[1], self.target[0])
        diff = abs(self.wrap(theta - desired))
        return -diff

    # 🔥 draw green target line
    def _draw_target(self):
        if self.marker_id is not None:
            p.removeUserDebugItem(self.marker_id)
        x, y = self.target
        self.marker_id = p.addUserDebugLine(
            [x, y, 0],
            [x, y, 0.2],
            [0, 1, 0],   # green
            3
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self._make_arm()
        self.target = self._new_target()
        self.steps = 0

        if self.render:
            self._draw_target()

        return self._get_state(), {}

    def step(self, action):
        self.steps += 1

        # velocity control
        vel = float(action[0]) * 3.0
        p.setJointMotorControl2(
            self.arm,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=vel,
            force=5
        )

        p.stepSimulation()

        state = self._get_state()
        theta = state[0]
        reward = self._compute_reward(theta)

        done = False
        truncated = False
        if self.steps > 200:
            truncated = True

        if self.render:
            self._draw_target()

        return state, reward, done, truncated, {}
