from stable_baselines3 import PPO
from arm_env import TwoJointArmEnv
import time

env = TwoJointArmEnv(render=True)
model = PPO.load("torque_arm_final", env=env)

obs, _ = env.reset()


while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1/240)

    if terminated or truncated:
        obs, _ = env.reset()
