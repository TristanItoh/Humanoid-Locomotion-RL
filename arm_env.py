import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import random

class ArmEnv(gym.Env):
    """
    Single-joint arm environment with delta-angle RL control + low-level PID torque control.
    
    Architecture:
    - RL Policy: outputs delta_theta (desired angle change per step)
    - PID Controller: converts target angle to motor torque
    - Physics: torque → angular acceleration → motion
    """
    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render=False, control_freq=100):
        super().__init__()

        self.render = render
        self.control_freq = control_freq  # Hz
        self.dt = 1.0 / control_freq  # timestep duration
        
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, 0)  # No gravity for now

        # ═══════════════════════════════════════════════════════════
        # ACTION SPACE: Delta angle (incremental change in target angle)
        # Alternative: Set to direct angle mode by uncommenting below
        # ═══════════════════════════════════════════════════════════
        self.use_delta_angle = True  # Toggle between delta and direct mode
        self.max_delta_angle = 0.2  # radians per timestep (~11 degrees)
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # ═══════════════════════════════════════════════════════════
        # OBSERVATION SPACE: [theta, omega, target_angle, target_x, target_y]
        # ═══════════════════════════════════════════════════════════
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10.0, -np.pi, -1.0, -1.0]),
            high=np.array([np.pi, 10.0, np.pi, 1.0, 1.0]),
            dtype=np.float32
        )

        # ═══════════════════════════════════════════════════════════
        # PID CONTROLLER GAINS
        # ═══════════════════════════════════════════════════════════
        self.kp = 10.0   # Proportional gain
        self.kd = 2.0    # Derivative gain
        self.ki = 0.1    # Integral gain
        self.max_torque = 5.0  # Maximum motor torque (Nm)
        
        # PID state
        self.integral_error = 0.0
        self.prev_error = 0.0

        # ═══════════════════════════════════════════════════════════
        # PHYSICAL PARAMETERS
        # ═══════════════════════════════════════════════════════════
        self.L = 1.0  # Arm length
        self.target_angle = 0.0  # Current target angle for PID controller
        
        self._make_arm()
        self.target = None
        self.marker_id = None
        self.steps = 0
        self.max_steps = 500

    def _make_arm(self):
        """Create single-joint arm with realistic inertia."""
        L = self.L

        # Fixed base at origin
        self.base = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0]
        )

        # Arm link (red box)
        link_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L/2, 0.05, 0.05],
            collisionFramePosition=[L/2, 0, 0]
        )
        link_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L/2, 0.05, 0.05],
            visualFramePosition=[L/2, 0, 0],
            rgbaColor=[1, 0, 0, 1]
        )

        self.arm = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0],
            linkMasses=[1.0],  # 1 kg arm
            linkCollisionShapeIndices=[link_col],
            linkVisualShapeIndices=[link_vis],
            linkPositions=[[0, 0, 0]],
            linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1]]  # Rotation around Z axis
        )

        # Enable torque control (disable default motor)
        p.setJointMotorControl2(
            self.arm,
            jointIndex=0,
            controlMode=p.VELOCITY_CONTROL,
            force=0  # Disable default motor
        )

    def _new_target(self):
        """Generate random 2D target point."""
        r = random.uniform(0.3, 1.0)
        ang = random.uniform(-math.pi, math.pi)
        return [r * math.cos(ang), r * math.sin(ang)]

    def wrap_angle(self, angle):
        """Wrap angle to [-π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _get_state(self):
        """Get current observation: [theta, omega, target_angle, target_x, target_y]."""
        joint_state = p.getJointState(self.arm, 0)
        theta = self.wrap_angle(joint_state[0])  # Position
        omega = joint_state[1]  # Velocity
        
        return np.array([
            theta,
            omega,
            self.wrap_angle(self.target_angle),  # What angle we told PID to track
            self.target[0],
            self.target[1]
        ], dtype=np.float32)

    def _compute_pid_torque(self, current_angle, current_velocity):
        """
        Low-level PID controller: converts target angle to motor torque.
        
        This is the "classical control" layer that handles motor dynamics.
        """
        # Compute error
        error = self.wrap_angle(self.target_angle - current_angle)
        
        # Integral term (with anti-windup)
        self.integral_error += error * self.dt
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)
        
        # Derivative term (velocity error)
        derivative = -current_velocity  # We want zero velocity at target
        
        # PID formula
        torque = (
            self.kp * error +
            self.ki * self.integral_error +
            self.kd * derivative
        )
        
        # Clamp to maximum motor torque
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        
        return torque

    def _compute_reward(self, theta, omega):
        """
        Reward function that encourages:
        1. Minimizing angle error
        2. Reducing angular velocity (prevent oscillation)
        3. Bonus for staying aligned
        4. Penalty for setting wrong target angle
        """
        # Desired angle to point at target
        desired_angle = math.atan2(self.target[1], self.target[0])
        angle_error = abs(self.wrap_angle(theta - desired_angle))
        
        # NEW: Penalty for setting target_angle incorrectly
        target_angle_error = abs(self.wrap_angle(self.target_angle - desired_angle))
        
        # Reward components
        angle_reward = -angle_error  # Penalize angle error
        velocity_penalty = -0.01 * abs(omega)  # Small penalty for moving
        target_penalty = -0.5 * target_angle_error  # Penalize bad target selection
        
        # Bonus for being well-aligned and still
        alignment_bonus = 0.0
        if angle_error < 0.05 and abs(omega) < 0.1:  # ~3 degrees, slow
            alignment_bonus = 1.0
        
        reward = angle_reward + velocity_penalty + target_penalty + alignment_bonus
        
        return reward

    def _draw_target(self):
        """Draw green line at target position."""
        if self.marker_id is not None:
            p.removeUserDebugItem(self.marker_id)
        
        x, y = self.target
        self.marker_id = p.addUserDebugLine(
            [x, y, 0],
            [x, y, 0.3],
            [0, 1, 0],  # Green
            lineWidth=3
        )

    def reset(self, seed=None, options=None):
        """Reset environment and generate new target."""
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, 0)
        
        self._make_arm()
        self.target = self._new_target()
        self.steps = 0
        
        # Reset PID controller state
        self.target_angle = 0.0
        self.integral_error = 0.0
        self.prev_error = 0.0

        if self.render:
            self._draw_target()

        return self._get_state(), {}

    def step(self, action):
        """
        Execute one step of delta-angle control.
        
        Pipeline:
        1. RL policy outputs delta_angle
        2. Update target_angle for PID controller
        3. PID computes torque from angle error
        4. Apply torque to joint
        5. Physics simulation step
        6. Compute reward based on alignment
        """
        self.steps += 1

        # ═══════════════════════════════════════════════════════════
        # STEP 1: RL Policy Output (delta angle or direct angle)
        # ═══════════════════════════════════════════════════════════
        if self.use_delta_angle:
            delta_angle = float(action[0]) * self.max_delta_angle
            self.target_angle = self.wrap_angle(self.target_angle + delta_angle)
        else:
            # Direct angle mode: action directly sets target
            self.target_angle = float(action[0]) * np.pi  # Maps [-1,1] to [-π,π]

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Get current joint state
        # ═══════════════════════════════════════════════════════════
        joint_state = p.getJointState(self.arm, 0)
        current_angle = joint_state[0]
        current_velocity = joint_state[1]

        # ═══════════════════════════════════════════════════════════
        # STEP 3: PID Controller (low-level control)
        # ═══════════════════════════════════════════════════════════
        torque = self._compute_pid_torque(current_angle, current_velocity)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Apply torque to joint
        # ═══════════════════════════════════════════════════════════
        p.setJointMotorControl2(
            self.arm,
            jointIndex=0,
            controlMode=p.TORQUE_CONTROL,
            force=torque
        )

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Physics simulation
        # ═══════════════════════════════════════════════════════════
        p.stepSimulation()

        # ═══════════════════════════════════════════════════════════
        # STEP 6: Compute reward and check termination
        # ═══════════════════════════════════════════════════════════
        state = self._get_state()
        theta = state[0]
        omega = state[1]
        
        reward = self._compute_reward(theta, omega)

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        if self.render:
            self._draw_target()

        return state, reward, terminated, truncated, {}

    def close(self):
        """Clean up PyBullet connection."""
        p.disconnect()


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE TRAINING LOOP (using Stable-Baselines3)
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Uncomment to test with random actions
    env = ArmEnv(render=True)
    
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Random delta angles
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
    
    # To train with RL:
    # from stable_baselines3 import PPO
    # env = ArmEnv(render=False)
    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100000)
    # model.save("arm_delta_angle_policy")