import time
from stable_baselines3 import PPO
from arm_env import ArmEnv

env = ArmEnv(render=True)
model = PPO.load("arm_model")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, trunc, info = env.step(action)
    time.sleep(1/240)   # smooth skibidi motion
