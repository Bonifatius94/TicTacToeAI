"""This module provides a trainable agent that is capable of learning TicTacToe."""

from dataclasses import dataclass
import numpy as np

from tictactoe.core import TicTacToeAction, TicTacToeSide, CROSS, \
    TicTacToeState, TicTacToeExperience, create_action, invert_state, action_pos


@dataclass
class TrainableTicTacToeModel:
    """Representing a trainable TicTacToe model."""
    weights: np.ndarray = None
    biases: np.ndarray = None
    learn_rate: float = 0.01
    reward_discount: float = 0.9

    def __post_init__(self):
        if not self.weights:
            self.weights = np.random.normal(loc=0.0, scale=0.1, size=(9, 9))
        if not self.biases:
            self.biases = np.random.normal(loc=0.0, scale=0.1, size=(9))

    def train(self, exp: TicTacToeExperience):
        """Update the model weights training with the given experience."""
        inputs = np.array(exp.state_before.board) - 1
        pred = self.predict(exp.state_before)[0][action_pos(exp.action)]
        can_eval_est_1 = exp.state_after and not exp.state_after.is_game_over
        label = self.reward_discount * np.max(self.predict(exp.state_after)) \
                    if can_eval_est_1 else exp.reward

        d_w = -2.0 * (label - pred) * inputs
        d_b = -2.0 * (label - pred)
        d_w, d_b = np.clip(d_w, -1, 1), np.clip(d_b, -1, 1)
        self.weights[:, action_pos(exp.action)] -= d_w * self.learn_rate
        self.biases -= d_b * self.learn_rate

    def predict(self, state: TicTacToeState) -> np.ndarray:
        """Predict the action scores for the given state."""
        inputs = np.expand_dims(np.array(state.board), axis=0) - 1
        return np.matmul(inputs, self.weights) + self.biases


@dataclass
class TrainableTicTacToeAgent:
    """Representing a trainable TicTacToe agent."""
    side: TicTacToeSide
    model: TrainableTicTacToeModel = TrainableTicTacToeModel()
    is_trainable: bool = True
    expl_rate: float = 0.1

    def choose_action(self, state: TicTacToeState) -> TicTacToeAction:
        """Choose the best action"""
        pos: int = np.argmax(self.model.predict(self._norm_state(state)))
        pos = pos if np.random.uniform() > self.expl_rate else np.random.choice(range(9))
        return create_action(pos, self.side)

    def train(self, exp: TicTacToeExperience):
        """Update the model weights training with the given experience."""
        exp.state_before = self._norm_state(exp.state_before)
        exp.state_after = self._norm_state(exp.state_after) if exp.state_after else None
        self.model.train(exp)

    def _norm_state(self, state: TicTacToeState) -> TicTacToeState:
        return invert_state(state) if self.side == CROSS else state


@dataclass
class RandomTicTacToeAgent:
    """Representing a randomly acting TicTacToe agent."""
    side: TicTacToeSide
    is_trainable: bool = False

    def choose_action(self, _: TicTacToeState) -> TicTacToeAction:
        """Choose the best action"""
        return create_action(np.random.choice(range(9)), self.side)

    def train(self, _: TicTacToeExperience):
        """Nothing to do here. Agent is not trainable"""
        pass
