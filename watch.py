from stable_baselines3 import PPO
from arm_env import ArmEnv
import time

env = ArmEnv(render=True)
model = PPO.load("arm_model", env=env)

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/120)

    if terminated or truncated:
        obs, _ = env.reset()
