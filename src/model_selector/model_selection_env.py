import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gymnasium as gym
from gymnasium import spaces

from model_selector.utils import calc_ks_score, blend_predictions
from model_selector.training import train_eval_model


class ModelSelectionEnv(gym.Env):
    """
    Reinforcement Learning Environment for Model Selection

    This environment implements the OpenAI Gymnasium interface for model selection tasks.
    It allows an RL agent to select between different machine learning models and evaluate
    their performance on a validation dataset.

    The environment supports 5 possible actions:
    - 0: XGBoost
    - 1: LightGBM
    - 2: Random Forest
    - 3: Deep Neural Network
    - 4: Blend of all models

    Each model is evaluated based on AUC and KS metrics, with penalties applied
    to more complex models to balance performance and computational cost.

    Attributes:
        device (str): Device to use for training ('cpu' or 'cuda')
        X_train (DataFrame): Training features
        X_val (DataFrame): Validation features
        y_train (Series): Training labels
        y_val (Series): Validation labels
        state (ndarray): Current state representation (feature means and variances)
        action_space (spaces.Discrete): The action space with 5 discrete actions
        observation_space (spaces.Box): The observation space for state representation
        terminated (bool): Whether the episode has terminated
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, X, y, device="cpu"):
        """
        Initialize the model selection environment.

        Args:
            X (DataFrame): Features dataset
            y (Series): Target labels
            device (str, optional): Device to use for training ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        super().__init__()
        self.device = device

        # Train/val split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.3, random_state=123
        )

        means = X.mean().values
        vars_ = X.var().values
        self.state = np.concatenate([means, vars_])  # observation

        # 5 discrete actions for 5 models
        self.action_space = spaces.Discrete(5)
        # states are the mean and variance of the features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.state),), dtype=np.float32
        )
        self.terminated = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            options (dict, optional): Additional options for reset. Defaults to None.

        Returns:
            tuple: Contains:
                - state (ndarray): The initial state observation
                - info (dict): Additional information (empty dictionary)
        """
        super().reset(seed=seed)
        self.terminated = False
        return self.state.astype(np.float32), {}

    def step(self, action):
        """
        Take a step in the environment by selecting and evaluating a model.

        Args:
            action (int): The action to take (0-4 corresponding to different models)

        Returns:
            tuple: Contains:
                - state (ndarray): The current state observation
                - reward (float): The reward for the action taken
                - terminated (bool): Whether the episode has terminated
                - truncated (bool): Whether the episode was truncated (always False)
                - info (dict): Additional information about the action and performance
        """
        if self.terminated:
            return self.state.astype(np.float32), 0.0, True, False, {}

        model_names = ["xgb", "lgbm", "rf", "dnn"]
        if action < 4:
            chosen_model = model_names[action]
            _, auc_v, ks_v, _ = train_eval_model(
                chosen_model,
                self.X_train,
                self.y_train,
                self.X_val,
                self.y_val,
                device=self.device,
            )
            penalty = 0.05 if chosen_model == "dnn" else 0.0
            reward = (auc_v + ks_v) - penalty
            info = {
                "action": action,
                "action_name": chosen_model,
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty,
            }
        else:
            # Blend
            probs_list = []
            for m in model_names:
                _, auc_m, ks_m, prob_m = train_eval_model(
                    m,
                    self.X_train,
                    self.y_train,
                    self.X_val,
                    self.y_val,
                    device=self.device,
                )
                probs_list.append(prob_m)
            final_prob = blend_predictions(probs_list)
            auc_v = roc_auc_score(self.y_val, final_prob)
            ks_v = calc_ks_score(self.y_val, final_prob)
            penalty = 0.1
            reward = (auc_v + ks_v) - penalty
            info = {
                "action": action,
                "action_name": "blend",
                "AUC": auc_v,
                "KS": ks_v,
                "Penalty": penalty,
            }

        self.terminated = True
        return self.state.astype(np.float32), reward, True, False, info
