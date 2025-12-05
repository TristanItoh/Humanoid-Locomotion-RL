from stable_baselines3 import PPO
from arm_env import ArmEnv

env = ArmEnv(render=False)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=50_000)
model.save("arm_model")
env.close()
