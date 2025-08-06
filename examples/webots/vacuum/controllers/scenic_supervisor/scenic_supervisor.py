from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.webots import WebotsSimulator


import gymnasium as gym
import numpy as np


from controller import Supervisor


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C,PPO


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy




import matplotlib.pyplot as plt
import time
import gc


start = time.time()
supervisor = Supervisor() # Collect the Supervisor node from the simulation
prefix = scenic.__file__[:-22]


simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),    
    "sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
    "position": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
})
def choose_difficulty(coverage):
    if coverage < 20:
        return "easy"
    elif coverage < 60:
        return "medium"
    else:
        return "hard"


def feedback_fn(result):
    """
    Feedback function for VerifAI CE sampler.
    Returns (1 - coverage) so CE focuses on scenes where the vacuum performs worst.
    """
    print("feedback_fn called")
   
    coverage = result.records["coverage"]
    return coverage[1]

                             
max_steps = 10000
episodes  = 10
difficulty = "easy"
model = None
total_timesteps = max_steps * episodes
episode_rewards = []
difficulty_log = []

for ep in range(episodes):


    print(f"\n=== EPISODE {ep + 1} | DIFFICULTY: {difficulty.upper()} ===")

    params = {
        "is_easy": difficulty == "easy",
        "is_medium": difficulty == "medium",
        "is_hard": difficulty == "hard"
    }
    scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                    model="scenic.simulators.webots.model",
                                    mode2D=False,
                                    params=params) 
               


    env = Monitor(ScenicGymEnv(scenario,
                    simulator,
                    render_mode=None,
                    max_steps=max_steps,
                    action_space=action_space,
                    observation_space=observation_space,
                    feedback_fn=feedback_fn)) # max_step is max step for an episode - Create an enviroment instance




    if ep==0:
        model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=0.0002, ent_coef=0.05)
    else:
        model = PPO.load("PPO_vacuum_final", env=env)

    model.learn(total_timesteps=max_steps)

    episodic_rewards = env.get_episode_rewards()
    rewards = env.get_episode_rewards()
    last_reward = rewards[-1] if rewards else 0
    episode_rewards.append(last_reward)
    coverage = env.env.get_coverage()
    print(f"The coverage is {coverage:.2f}%")
    difficulty = choose_difficulty(coverage)
    difficulty_log.append(difficulty)


    del model
    del env
    gc.collect()




# Plot reward progression
fig, ax = plt.subplots()
ax.stem(range(1, len(episode_rewards) + 1), episode_rewards)
ax.set_title("Episode Rewards (Coverage Proxy)")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
plt.savefig("PPO_curriculum_rewards.png")
plt.show()

# Save a second figure with proper limits
fig, ax = plt.subplots()
ax.scatter(range(len(episodic_rewards)), episodic_rewards)
ax.set_title("Episodic Rewards Scatter Plot")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_xlim(0, len(episodic_rewards))
ax.set_ylim(0, max(episodic_rewards) + 10)
file_name = "MLP_policy_" + str(total_timesteps) + ".png"
plt.savefig(file_name, format='png')
plt.show()




# ---------------- Final Evaluation ----------------
# Reload final environment (e.g., using last difficulty)
params = {
    "is_easy": difficulty == "easy",
    "is_medium": difficulty == "medium",
    "is_hard": difficulty == "hard"
}
scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                    model="scenic.simulators.webots.model",
                                    mode2D=False,
                                    params=params) 
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
