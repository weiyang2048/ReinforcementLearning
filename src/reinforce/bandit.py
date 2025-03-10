import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from model_selector.training import train_eval_model
from model_selector.utils import calc_ks_score, blend_predictions
import torch


def evaluate_action(action, X_train, X_val, y_train, y_val, device="cpu"):
    """
    Evaluate a model selection action and return the reward and performance metrics.

    Args:
        action (int): The action index to evaluate (0-3 for individual models, 4 for blend)
        X_train (DataFrame): Training features
        X_val (DataFrame): Validation features
        y_train (Series): Training labels
        y_val (Series): Validation labels
        device (str, optional): Device to use for training ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        tuple: Contains:
            - reward (float): The calculated reward for the action
            - auc_val (float): AUC score on validation data
            - ks_val (float): KS score on validation data
    """
    model_names = ["xgb", "lgbm", "rf", "dnn"]
    if action < 4:
        chosen_model = model_names[action]
        _, auc_val, ks_val, _ = train_eval_model(
            chosen_model, X_train, y_train, X_val, y_val, device=device
        )
        penalty = 0.05 if chosen_model == "dnn" else 0.0
        reward = (auc_val + ks_val) - penalty
        return reward, auc_val, ks_val
    else:
        # Blend
        probs_list = []
        for m in model_names:
            _, auc_m, ks_m, p = train_eval_model(
                m, X_train, y_train, X_val, y_val, device=device
            )
            probs_list.append(p)
        final_prob = blend_predictions(probs_list)
        auc_blend = roc_auc_score(y_val, final_prob)
        ks_blend = calc_ks_score(y_val, final_prob)
        reward = (auc_blend + ks_blend) - 0.1
        return reward, auc_blend, ks_blend


def multi_armed_bandit_model_selection(
    data,
    n_episodes=50,
    n_actions=5,
    epsilon=0.06,
    device="cpu",
):
    """
    Implement a multi-armed bandit algorithm for model selection.

    This function uses an epsilon-greedy strategy to explore different model options
    and exploit the best performing ones over multiple episodes.

    Args:
        data (DataFrame): The dataset containing features and 'label' column
        n_episodes (int, optional): Number of episodes to run. Defaults to 50.
        n_actions (int, optional): Number of possible actions (models). Defaults to 5.
        epsilon (float, optional): Exploration rate. Defaults to 0.06.
        device (str, optional): Device to use for training. Defaults to 'cpu'.

    Returns:
        tuple: Contains:
            - Q (ndarray): Final Q-values for each action
            - action_history (list): History of actions taken
            - reward_history (list): History of rewards received
            - action_auc_records (list): AUC scores for each action
            - action_ks_records (list): KS scores for each action
            - action_reward_records (list): Reward records for each action
    """
    Q = np.zeros(n_actions, dtype=np.float32)
    counts = np.zeros(n_actions, dtype=int)

    action_auc_records = [[] for _ in range(n_actions)]
    action_ks_records = [[] for _ in range(n_actions)]
    action_reward_records = [[] for _ in range(n_actions)]

    action_history = []
    reward_history = []

    for episode in range(n_episodes):
        seed = 1000 + episode
        X = data.drop("label", axis=1)  # Features
        y = data["label"]  # Labels

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=123
        )

        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            action = np.argmax(Q)

        reward, auc_val, ks_val = evaluate_action(
            action, X_train, X_val, y_train, y_val, device=device
        )

        counts[action] += 1
        Q[action] += (reward - Q[action]) / counts[action]

        action_history.append(action)
        reward_history.append(reward)
        action_auc_records[action].append(auc_val)
        action_ks_records[action].append(ks_val)
        action_reward_records[action].append(reward)

        print(
            f"Episode {episode+1}/{n_episodes}, "
            f"Action={action}, Reward={reward:.4f}, Updated Q={Q}"
        )

    return (
        Q,
        action_history,
        reward_history,
        action_auc_records,
        action_ks_records,
        action_reward_records,
    )


def run_bandit(data):
    """
    Run the multi-armed bandit algorithm and display results.

    Args:
        data (DataFrame): The dataset containing features and 'label' column

    Returns:
        None: Results are printed to console
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device={device}")

    n_episodes = 500
    n_actions = 5
    epsilon = 0.05

    (Q, actions, rewards, auc_records, ks_records, reward_records) = (
        multi_armed_bandit_model_selection(
            data=data,
            n_episodes=n_episodes,
            n_actions=n_actions,
            epsilon=epsilon,
            device=device,
        )
    )

    best_action = np.argmax(Q)
    model_names = ["XGB", "LightGBM", "RandomForest", "DNN", "Blend"]

    print("\n========================================")
    print("Interpreting Your Current Results")
    print("========================================\n")

    print("Final Q-values:", Q)
    print(f"Best action index: {best_action}")
    print(
        f"Best action is: {model_names[best_action]} with estimated Q = {Q[best_action]:.4f}\n"
    )

    print("Detailed AUC/KS/Reward by action:")
    print("--------------------------------------------------")
    for a in range(n_actions):
        if len(auc_records[a]) > 0:
            avg_auc = np.mean(auc_records[a])
            avg_ks = np.mean(ks_records[a])
            avg_reward = np.mean(reward_records[a])
            print(f"Action {a} ({model_names[a]}): chosen {len(auc_records[a])} times")
            print(
                f"  Mean AUC = {avg_auc:.4f}, Mean KS = {avg_ks:.4f}, Mean Reward = {avg_reward:.4f}\n"
            )
        else:
            print(f"Action {a} ({model_names[a]}): chosen 0 times\n")


if __name__ == "__main__":
    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    import boto3

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Download file from S3
    bucket_name = "weiy-bucket"
    file_name = "rein_data_binary.csv"
    try:
        s3_client.download_file(bucket_name, file_name, file_name)
        print(f"Successfully downloaded {file_name} from S3")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")

    data = pd.read_csv("rein_data_binary.csv")
    X = data.drop("label", axis=1)
    y = data["label"]

    os.remove(file_name)
    run_bandit(data)
