import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import random
import torch

class TwoJointArmEnv(gym.Env):
    """
    Two-joint arm environment with DIRECT TORQUE CONTROL.
    
    Architecture:
    - RL Policy: outputs [torque1, torque2] directly to motors
    - No PID layer, no target angles - pure motor control
    - Physics: torque → angular acceleration → motion
    - Goal: End-effector reaches 2D target point
    
    The network must learn:
    1. Forward kinematics (where am I pointing?)
    2. Inverse dynamics (what torques reach the target?)
    3. Momentum management (when to accelerate/brake)
    4. Coordinated control (both joints working together)
    
    This is the most realistic simulation of motor control.
    """
    metadata = {"render_modes": ["human"], "render_fps": 240}

    def __init__(self, render=False, control_freq=100, curriculum_stage=1.0):
        super().__init__()

        self.render = render
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.curriculum_stage = curriculum_stage
        
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, 0)  # No gravity for now

        # ═══════════════════════════════════════════════════════════
        # ACTION SPACE: [torque1, torque2]
        # Direct motor torque commands in Newton-meters
        # ═══════════════════════════════════════════════════════════
        self.max_torque = 5.0  # Maximum torque per joint (Nm)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # ═══════════════════════════════════════════════════════════
        # OBSERVATION SPACE:
        # [theta1, omega1, theta2, omega2, target_x, target_y]
        # ═══════════════════════════════════════════════════════════
        self.observation_space = spaces.Box(
            low=np.array([
                -np.pi, -10.0,   # Joint 1: angle, velocity
                -np.pi, -10.0,   # Joint 2: angle, velocity
                -2.0, -2.0       # Target position
            ]),
            high=np.array([
                np.pi, 10.0,
                np.pi, 10.0,
                2.0, 2.0
            ]),
            dtype=np.float32
        )

        # ═══════════════════════════════════════════════════════════
        # PHYSICAL PARAMETERS
        # ═══════════════════════════════════════════════════════════
        self.L1 = 0.6  # Length of first link (shoulder to elbow)
        self.L2 = 0.6  # Length of second link (elbow to end-effector)
        
        self._make_arm()
        self.target = None
        self.marker_id = None
        self.end_effector_marker = None
        self.steps = 0
        self.max_steps = 500

    def _make_arm(self):
        """Create two-joint arm with realistic inertia."""
        L1, L2 = self.L1, self.L2

        # Fixed base at origin
        self.base = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0]
        )

        # Link 1 (shoulder to elbow) - RED
        link1_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L1/2, 0.04, 0.04],
            collisionFramePosition=[L1/2, 0, 0]
        )
        link1_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L1/2, 0.04, 0.04],
            visualFramePosition=[L1/2, 0, 0],
            rgbaColor=[1, 0, 0, 1]
        )

        # Link 2 (elbow to end-effector) - BLUE
        link2_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[L2/2, 0.04, 0.04],
            collisionFramePosition=[L2/2, 0, 0]
        )
        link2_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[L2/2, 0.04, 0.04],
            visualFramePosition=[L2/2, 0, 0],
            rgbaColor=[0, 0, 1, 1]
        )

        # Create multi-body with realistic masses and inertia
        self.arm = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0, 0, 0],
            
            linkMasses=[1.0, 0.5],  # Realistic masses (kg)
            linkCollisionShapeIndices=[link1_col, link2_col],
            linkVisualShapeIndices=[link1_vis, link2_vis],
            linkPositions=[
                [0, 0, 0],      # Link 1 starts at base
                [L1, 0, 0]      # Link 2 starts at end of Link 1
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1]
            ],
            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],
            linkParentIndices=[0, 1],  # Link 1 → base, Link 2 → Link 1
            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 0, 1], [0, 0, 1]]  # Both rotate around Z
        )

        # Disable default velocity control - we use pure torque
        for joint_idx in range(2):
            p.setJointMotorControl2(
                self.arm,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                force=0  # Completely disable built-in motor
            )

    def _new_target(self, curriculum_stage=1.0):
        """
        Generate random 2D target point within reach.
        
        Curriculum learning:
        - Early: Bias toward far targets (forces learning extension)
        - Late: Full workspace
        """
        max_reach = self.L1 + self.L2
        min_reach = abs(self.L1 - self.L2)
        
        if curriculum_stage < 0.4:
            # Early: prefer far targets (70% far, 30% medium)
            if random.random() < 0.7:
                r = random.uniform(max_reach * 0.7, max_reach - 0.05)
            else:
                r = random.uniform(max_reach * 0.4, max_reach * 0.7)
        else:
            # Later: full workspace
            r = random.uniform(min_reach + 0.1, max_reach - 0.05)
        
        ang = random.uniform(-math.pi, math.pi)
        return [r * math.cos(ang), r * math.sin(ang)]

    def wrap_angle(self, angle):
        """Wrap angle to [-π, π]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _get_end_effector_pos(self):
        """
        Calculate end-effector position using forward kinematics.
        This is the tip of the second link.
        """
        # Get joint angles
        theta1 = p.getJointState(self.arm, 0)[0]
        theta2 = p.getJointState(self.arm, 1)[0]
        
        # Forward kinematics
        x1 = self.L1 * math.cos(theta1)
        y1 = self.L1 * math.sin(theta1)
        
        # theta2 is relative to link1, so absolute angle is theta1 + theta2
        x2 = x1 + self.L2 * math.cos(theta1 + theta2)
        y2 = y1 + self.L2 * math.sin(theta1 + theta2)
        
        return x2, y2

    def _get_state(self):
        """
        Get current observation:
        [theta1, omega1, theta2, omega2, target_x, target_y]
        
        Simple and direct - just joint states and target position.
        The network must learn everything else.
        """
        # Joint 1 (shoulder)
        joint1_state = p.getJointState(self.arm, 0)
        theta1 = self.wrap_angle(joint1_state[0])
        omega1 = joint1_state[1]
        
        # Joint 2 (elbow)
        joint2_state = p.getJointState(self.arm, 1)
        theta2 = self.wrap_angle(joint2_state[0])
        omega2 = joint2_state[1]
        
        return np.array([
            theta1, omega1,
            theta2, omega2,
            self.target[0], self.target[1]
        ], dtype=np.float32)

    def _compute_reward(self):
        """
        Reward function for torque control.
        
        Key components:
        1. Distance to target (main objective)
        2. Progress reward (moving toward target)
        3. Success bonus (reaching target)
        4. Energy penalty (efficient torque usage)
        5. Mild smoothness reward
        """
        # Get end-effector position
        ee_x, ee_y = self._get_end_effector_pos()
        
        # Current distance to target
        current_distance = math.sqrt(
            (ee_x - self.target[0])**2 + 
            (ee_y - self.target[1])**2
        )
        
        # Get joint velocities
        omega1 = p.getJointState(self.arm, 0)[1]
        omega2 = p.getJointState(self.arm, 1)[1]
        
        # ═══════════════════════════════════════════════════════════
        # COMPONENT 1: Distance to target
        # ═══════════════════════════════════════════════════════════
        distance_reward = -current_distance * 10.0
        
        # ═══════════════════════════════════════════════════════════
        # COMPONENT 2: Progress reward (critical for learning!)
        # ═══════════════════════════════════════════════════════════
        progress_reward = 0.0
        if hasattr(self, 'prev_distance') and self.prev_distance is not None:
            delta_distance = self.prev_distance - current_distance
            progress_reward = delta_distance * 50.0
        
        self.prev_distance = current_distance
        
        # ═══════════════════════════════════════════════════════════
        # COMPONENT 3: Success bonus
        # ═══════════════════════════════════════════════════════════
        success_bonus = 0.0
        if current_distance < 0.05:  # Within 5cm
            success_bonus = 20.0
        elif current_distance < 0.1:  # Within 10cm
            success_bonus = 5.0
        
        # ═══════════════════════════════════════════════════════════
        # COMPONENT 4: Energy efficiency
        # Penalize excessive torque usage (like real robots)
        # ═══════════════════════════════════════════════════════════
        if hasattr(self, 'last_torques'):
            torque1, torque2 = self.last_torques
            energy_penalty = -0.001 * (abs(torque1) + abs(torque2))
        else:
            energy_penalty = 0.0
        
        # ═══════════════════════════════════════════════════════════
        # COMPONENT 5: Smoothness (velocity penalty)
        # ═══════════════════════════════════════════════════════════
        velocity_penalty = -0.005 * (abs(omega1) + abs(omega2))
        
        # ═══════════════════════════════════════════════════════════
        # TOTAL REWARD
        # ═══════════════════════════════════════════════════════════
        total_reward = (
            distance_reward +
            progress_reward +
            success_bonus +
            energy_penalty +
            velocity_penalty
        )
        
        return total_reward

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

    def _draw_end_effector(self):
        """Draw yellow line at end-effector."""
        if self.end_effector_marker is not None:
            p.removeUserDebugItem(self.end_effector_marker)
        
        ee_x, ee_y = self._get_end_effector_pos()
        self.end_effector_marker = p.addUserDebugLine(
            [ee_x, ee_y, 0],
            [ee_x, ee_y, 0.15],
            [1, 1, 0],  # Yellow
            lineWidth=5
        )

    def reset(self, seed=None, options=None):
        """Reset environment and generate new target."""
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, 0)
        
        self._make_arm()
        
        # Initialize arm in random pose
        init_theta1 = random.uniform(-math.pi, math.pi)
        init_theta2 = random.uniform(-math.pi/2, math.pi/2)
        
        # Set initial joint positions with zero velocity
        p.resetJointState(self.arm, 0, init_theta1, 0.0)
        p.resetJointState(self.arm, 1, init_theta2, 0.0)
        
        self.target = self._new_target(self.curriculum_stage)
        self.steps = 0
        
        # Reset tracking variables
        self.prev_distance = None
        self.last_torques = [0.0, 0.0]

        if self.render:
            self._draw_target()

        return self._get_state(), {}

    def step(self, action):
        """
        Execute one step with direct torque control.
        
        Pipeline:
        1. RL policy outputs [torque1, torque2] (normalized to [-1, 1])
        2. Scale to actual torque values
        3. Apply torques directly to joints
        4. Physics simulation step
        5. Compute reward based on end-effector position
        
        No PID, no target angles - pure motor control!
        """
        self.steps += 1

        # ═══════════════════════════════════════════════════════════
        # STEP 1 & 2: Get torque commands and scale
        # ═══════════════════════════════════════════════════════════
        torque1 = float(action[0]) * self.max_torque
        torque2 = float(action[1]) * self.max_torque
        
        self.last_torques = [torque1, torque2]

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Apply torques directly to motors
        # ═══════════════════════════════════════════════════════════
        p.setJointMotorControl2(
            self.arm,
            jointIndex=0,
            controlMode=p.TORQUE_CONTROL,
            force=torque1
        )
        
        p.setJointMotorControl2(
            self.arm,
            jointIndex=1,
            controlMode=p.TORQUE_CONTROL,
            force=torque2
        )

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Physics simulation
        # ═══════════════════════════════════════════════════════════
        p.stepSimulation()

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Get new state and compute reward
        # ═══════════════════════════════════════════════════════════
        state = self._get_state()
        reward = self._compute_reward()

        terminated = False
        truncated = False
        if self.steps >= self.max_steps:
            truncated = True

        if self.render:
            self._draw_target()
            self._draw_end_effector()

        return state, reward, terminated, truncated, {}

    def close(self):
        """Clean up PyBullet connection."""
        p.disconnect()


# ═══════════════════════════════════════════════════════════════════
# TRAINING SCRIPT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        print("="*60)
        print("TRAINING: Two-Joint Arm with Direct Torque Control")
        print("="*60)
        print("\nThis is the most realistic motor control simulation.")
        print("The network must learn:")
        print("  - Forward kinematics (where does my hand go?)")
        print("  - Inverse dynamics (what torques reach the target?)")
        print("  - Momentum management (when to push/brake)")
        print("  - Energy efficiency (don't waste torque)")
        print()
        
        start_time = time.time()
        
        # Create environment
        env = TwoJointArmEnv(render=False, control_freq=100, curriculum_stage=0.0)
        
        # Create PPO model with larger network (torque control is harder!)
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # 3 layers for complex control
                activation_fn=torch.nn.ReLU
            )
        )
        
        # Stage 1: Learn with far targets
        print("\n" + "="*60)
        print("STAGE 1: Learning to reach far targets")
        print("="*60)
        env.curriculum_stage = 0.0
        model.learn(total_timesteps=3_000_000)
        model.save("torque_arm_stage1")
        
        elapsed = (time.time() - start_time) / 60
        print(f"\nStage 1 complete! Time: {elapsed:.1f} minutes")
        
        # Stage 2: Full workspace
        print("\n" + "="*60)
        print("STAGE 2: Full workspace training")
        print("="*60)
        env.curriculum_stage = 1.0
        model.learn(total_timesteps=6_000_000)
        model.save("torque_arm_final")
        
        total_elapsed = (time.time() - start_time) / 3600
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE! Total time: {total_elapsed:.2f} hours")
        print(f"Total timesteps: 9,000,000")
        print("="*60)
        print("\nModel saved as 'torque_arm_final'")
        print("\nTo test: python script.py")
        
    else:
        # Visualization mode
        print("="*60)
        print("VISUALIZATION MODE")
        print("="*60)
        print("\nControls:")
        print("  - Watch the arm reach for green targets")
        print("  - Yellow line = end-effector position")
        print("  - Press Ctrl+C to stop")
        print()
        print("To train: python script.py train")
        print()
        
        env = TwoJointArmEnv(render=True, curriculum_stage=1.0)
        
        # Try to load trained model
        try:
            from stable_baselines3 import PPO
            model = PPO.load("torque_arm_final")
            print("✓ Loaded trained model: torque_arm_final")
            use_model = True
        except:
            print("✗ No trained model found - using random actions")
            print("  Train first with: python script.py train")
            use_model = False
        
        obs, info = env.reset()
        
        try:
            episode = 0
            while True:
                if use_model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    episode += 1
                    if episode % 10 == 0:
                        print(f"Completed {episode} episodes...")
                    obs, info = env.reset()
                    
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        env.close()