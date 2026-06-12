import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import ale_py

print("Initializing Gymnasium environment for training...")

# Create the raw Gymnasium environment (without 'human' rendering to keep training fast)
train_env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

# Apply SB3's Atari wrapper (handles 84x84 scaling, grayscaling, life loss terminal states)
train_env = AtariWrapper(train_env)

# SB3 algorithms expect a vectorized environment format (VecEnv)
train_env = DummyVecEnv([lambda: train_env])

# Stack 4 frames so the network can perceive motion and ball direction
train_env = VecFrameStack(train_env, n_stack=4)

# Define the Deep Q-Network
model = DQN("CnnPolicy", train_env, verbose=1, buffer_size=10000, learning_rate=1e-4)

# Train the model (Keep timesteps low for a quick lab test, increase to 100,000+ for real results)
print("Starting short training run...")
model.learn(total_timesteps=100000)
print("Training complete! Initializing visualization demo...")


# Create a fresh Gymnasium environment explicitly configured to render visually
eval_env = gym.make("ALE/Breakout-v5", render_mode="human")

# Apply the identical wrappers so the input frame dimensions match what the CNN learned
eval_env = AtariWrapper(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])
eval_env = VecFrameStack(eval_env, n_stack=4)

# Run a live test episode
obs = eval_env.reset()
done = False

print("Watch the game window! Press Ctrl+C in the terminal to close.")

try:
    while True:
        # Use the trained model to predict the next action
        action, _states = model.predict(obs, deterministic=True)
        
        # Execute the action in the visual environment
        obs, rewards, dones, infos = eval_env.step(action)
except KeyboardInterrupt:
    print("Keyboard interrupt")        


eval_env.close()
print("Demo finished successfully!")