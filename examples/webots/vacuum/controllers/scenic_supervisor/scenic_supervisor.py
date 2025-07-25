from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.webots import WebotsSimulator
from controller import Supervisor

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import gc

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------- Setup ----------------
start = time.time()

supervisor = Supervisor()
simulator = WebotsSimulator(supervisor)

prefix = scenic.__file__[:-22]
scenic_file = prefix + "examples/webots/vacuum/vacuum.scenic"

action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,), dtype=np.float64),
    "sensor": gym.spaces.Box(low=np.zeros(7), high=np.ones(7), shape=(7,), dtype=np.float64),
    "position": gym.spaces.Box(low=np.array([-2.6, -2.6]), high=np.array([2.6, 2.6]), shape=(2,), dtype=np.float64)
})

# ---------------- Curriculum Function ----------------
def choose_difficulty(coverage):
    if coverage < 20:
        return "easy"
    elif coverage < 60:
        return "medium"
    else:
        return "hard"
# ---------------- Feedback Function ----------------

def feedback_fn(result):
    """
    Feedback function for VerifAI CE sampler.
    Returns (1 - coverage) so CE focuses on scenes where the vacuum performs worst.
    """
    print("feedback_fn called")
    
    coverage = result.records["coverage"]
    return coverage[1]
    

# # ---------------- Training Loop ----------------
episodes = 10
max_steps = 10000
difficulty = "easy"
model = None

episode_rewards = []
difficulty_log = []

for ep in range(episodes):
    print(f"\n=== EPISODE {ep + 1} | DIFFICULTY: {difficulty.upper()} ===")
    # Pass difficulty to Scenic
    params = {
        "is_easy": difficulty == "easy",
        "is_medium": difficulty == "medium",
        "is_hard": difficulty == "hard"
    }

    # Load Scenic scenario with current difficulty
    scenario = scenic.scenarioFromFile(
        scenic_file,
        mode2D=False,
        params=params
    )

    # Create environment
    env = Monitor(ScenicGymEnv(
        scenario,
        simulator,
        render_mode=None,
        max_steps=max_steps,
        action_space=action_space,
        observation_space=observation_space,
        feedback_fn =feedback_fn ))
    # model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=0.0002, ent_coef=0.05)
    # model = PPO.load("PPO_vacuum_final", env=env)
    # model.set_parameters("PPO_vacuum_agent")
    if ep==0:
        model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=0.0002, ent_coef=0.05)
    else:
        model = PPO.load("PPO_vacuum_final", env=env)
    #     model.set_parameters("PPO_vacuum_agent")
        

    # Load or reuse model
    

    # Train for one episode
    print("starting learning")
    model.learn(total_timesteps=max_steps)
    print("finish learning")

    # Evaluate and track reward
    rewards = env.get_episode_rewards()
    last_reward = rewards[-1] if rewards else 0
    episode_rewards.append(last_reward)

    # Estimate coverage and update difficulty
# Make sure these are coming from the environment
    coverage = env.env.get_coverage()
    print(f"The coverage is {coverage:.2f}%")
    difficulty = choose_difficulty(coverage)
    difficulty_log.append(difficulty)

    # Clean up
    del env
    model.save("PPO_vacuum_final")

    del model
    gc.collect()
    


# ---------------- Save Model ----------------

# ---------------- Plot Reward ----------------
fig, ax = plt.subplots()
ax.stem(range(1, len(episode_rewards) + 1), episode_rewards)
ax.set_title("Episode Rewards (Coverage Proxy)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
plt.savefig("PPO_curriculum_rewards.png")
plt.show()

# ---------------- Final Evaluation ----------------
# Reload final environment (e.g., using last difficulty)
params = {
    "is_easy": difficulty == "easy",
    "is_medium": difficulty == "medium",
    "is_hard": difficulty == "hard"
}
scenario = scenic.scenarioFromFile(
    scenic_file,
    mode2D=False,
    params=params
)
env = Monitor(ScenicGymEnv(
    scenario,
    simulator,
    render_mode=None,
    max_steps=max_steps,
    action_space=action_space,
    observation_space=observation_space,
    feedback_fn=feedback_fn
))
model = PPO.load("PPO_vacuum_final", env=env)

mean_rwd, std_rwd = evaluate_policy(model, env, n_eval_episodes=10, render=False, deterministic=False)
print(f"\nFinal evaluation — Mean reward: {mean_rwd:.2f}, Std: {std_rwd:.2f}")
print(f"Total training time: {(time.time() - start) / 60:.2f} minutes")
