"""
Reinforcement Learning Model Selection Module

This module implements a Deep Q-Network (DQN) approach to automate model selection
for machine learning tasks. It uses a custom environment that simulates the process
of selecting the best model from a set of candidates based on performance metrics.

The module includes:
- A custom callback for monitoring training progress
- Functions to set up and train the DQN agent
- Evaluation of the final policy to determine the best model

Usage:
    python src/reinforce/train_rl_model.py -t <total_timesteps>

Args:
    -t, --total_timesteps: Total number of timesteps for training the model (default: 2000)
"""

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import argparse

from model_selector.model_selection_env import ModelSelectionEnv
from data_loader.data_loader import download_data_from_s3, prepare_data
from torch.utils.tensorboard import SummaryWriter
import datetime

writer = SummaryWriter(
    "runs/dqn_experiment" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)


class CustomCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.

    This callback logs metrics to TensorBoard during training, including
    exploration rate, rewards, and loss values.

    Attributes:
        writer: TensorBoard SummaryWriter for logging metrics
    """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.writer = writer  # Use the global writer

    def _on_step(self):
        """
        This method is called at each step of training.

        Logs current metrics to TensorBoard and prints progress information
        at regular intervals if verbose mode is enabled.

        Returns:
            bool: Whether to continue training or not
        """
        # Log exploration rate
        exploration_rate = self.model.exploration_rate
        self.writer.add_scalar("Exploration Rate", exploration_rate, self.num_timesteps)
        if self.num_timesteps > 5:
            # current_state = self.locals["obs"]
            # env = self.model.get_env()
            # reset the environment
            # obs = env.reset()
            # get current state
            # action, _ = self.model.predict(current_state, deterministic=False)
            # step the environment
            # obs, reward, dones, infos = env.step(action)
            infos = self.locals["infos"]
            auc_v = infos[0].get("AUC", [0])
            ks_v = infos[0].get("KS", [0])
            action = infos[0].get("action", [0])
            reward = self.locals["rewards"][0]
            self.writer.add_scalar("AUC", auc_v, self.num_timesteps)
            self.writer.add_scalar("KS", ks_v, self.num_timesteps)
            self.writer.add_scalar("Action", action, self.num_timesteps)
            self.writer.add_scalar("Reward", reward, self.num_timesteps)
            # Log loss if available
            if "loss" in self.model.logger.name_to_value:
                loss = self.model.logger.name_to_value["loss"]
                self.writer.add_scalar("Loss", loss, self.num_timesteps)

            if self.verbose > 0 and self.n_calls % 100 == 0:
                print(
                    f"Step: {self.n_calls}, Mean reward: {self.locals['rewards'][0]:.4f}, Exploration rate: {exploration_rate:.4f}"
                )

        return True


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run RL model selection with DQN.")
    parser.add_argument(
        "-t",
        "--total_timesteps",
        type=int,
        default=2000,
        help="Total number of timesteps for training the model.",
    )
    return parser.parse_args()


def run_rl_model_selection_pytorch(X, y, total_timesteps):
    """
    Run the reinforcement learning model selection process.

    Sets up the environment, trains a DQN agent, and evaluates the final policy
    to determine the best model for the given dataset.

    Args:
        X (DataFrame): Feature data
        y (Series): Target data
        total_timesteps (int): Total number of timesteps for training

    Returns:
        None: Prints the final selected model and its reward
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create single-step Gymnasium environment
    env = ModelSelectionEnv(X, y, device=device)

    # Wrap with DummyVecEnv
    def make_env():
        return env

    vec_env = DummyVecEnv([make_env])

    # Create callback
    callback = CustomCallback(verbose=1)

    # Create DQN
    model = DQN(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        # tensorboard_log="./rl_tensorboard/dqn_experiment",
    )

    # Train with callback
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # reset the environment
    obs = vec_env.reset()
    # get the action from the model
    action, _ = model.predict(obs, deterministic=True)
    # step the environment
    obs, rewards, dones, infos = vec_env.step(action)
    # get the final reward
    final_reward = rewards[0]
    # print the final action and reward
    action_map = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]
    print("\n======================================")
    print(f"Final chosen action => {action[0]} ({action_map[action[0]]})")
    print(f"Final step reward => (AUC + KS - penalty) = {final_reward:.4f}")
    print("======================================\n")


if __name__ == "__main__":
    # python src/reinforce/train_rl_model.py -t 10
    import warnings

    warnings.filterwarnings("ignore")
    data = download_data_from_s3("weiy-bucket", "rein_data_binary.csv")
    X, y = prepare_data(data)
    args = parse_arguments()
    run_rl_model_selection_pytorch(X, y, args.total_timesteps)
