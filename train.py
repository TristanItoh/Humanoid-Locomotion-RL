from stable_baselines3 import PPO
from arm_env import ArmEnv

env = ArmEnv(render=False, control_freq=100)

# Increase network size slightly since we added an observation
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    policy_kwargs=dict(net_arch=[128, 128])  # Slightly larger network
)

model.learn(total_timesteps=500000)
model.save("arm_policy_improved")