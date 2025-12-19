from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from arm_env import TwoJointArmEnv
import time

class CurriculumCallback(BaseCallback):
    """Gradually increases curriculum difficulty during training."""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        
    def _on_step(self):
        # Gradually increase from 0.0 to 1.0 throughout training
        progress = self.num_timesteps / self.total_timesteps
        self.training_env.envs[0].curriculum_stage = min(1.0, progress * 2.0)
        
        # Log every 500k steps
        if self.num_timesteps % 500_000 == 0:
            stage = self.training_env.envs[0].curriculum_stage
            print(f"\n[Progress] {self.num_timesteps:,} steps | Curriculum: {stage:.2f}")
        
        return True

# Create environment
env = TwoJointArmEnv(render=False, curriculum_stage=0.0)

# Create model
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    policy_kwargs=dict(net_arch=[256, 256, 128])  # Added 3rd layer for torque control
)

# Train with automatic curriculum
total_timesteps = 2_000_000  # ~5 hours
callback = CurriculumCallback(total_timesteps)

print("="*60)
print(f"Training for {total_timesteps:,} timesteps")
print("Curriculum will progress automatically from 0.0 to 1.0")
print("="*60)

start_time = time.time()
model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

# Save final model
model.save("torque_arm_final")

elapsed = (time.time() - start_time) / 3600
print(f"\n{'='*60}")
print(f"Training complete! Total time: {elapsed:.2f} hours")
print(f"Model saved as: torque_arm_final.zip")
print(f"{'='*60}")