"""This module provides a game session to make agents compete against each other."""

import random
from typing import Protocol, List, Tuple
from dataclasses import dataclass

from tictactoe.core import TicTacToeAction, TicTacToeSide, \
    TicTacToeState, TicTacToeEnv, TicTacToeExperience, opponent


class TicTacToeAgent(Protocol):
    """Representing a blueprint of a TicTacToe agent."""
    is_trainable: bool
    side: TicTacToeSide

    def choose_action(self, state: TicTacToeState) -> TicTacToeAction:
        """Select an action for the given state."""
        ...

    def train(self, exp: TicTacToeExperience):
        """Update the model to be trained."""
        ...


@dataclass
class TicTacToeSession:
    """Representing a TicTacToe game session where two agents play games
    against each other and learn from the experience gained."""
    player_1: TicTacToeAgent
    player_2: TicTacToeAgent
    env: TicTacToeEnv = TicTacToeEnv()
    invalid_draws_count: int = 0

    def play_game(self) -> Tuple[List[TicTacToeAction], TicTacToeSide]:
        """Let the agents play a single game of TicTacToe against each other."""
        state = self.env.reset()
        acting_side = random.choice([TicTacToeSide.CROSS, TicTacToeSide.CIRCLE])
        acting_player = lambda: self.player_1 \
            if self.player_1.side == acting_side else self.player_2

        actions = []
        states = [state]

        while not state.is_game_over:
            player = acting_player()

            action = self.select_action(player, state)
            state = self.env.apply_action(action)
            
            states.append(state)
            actions.append(action)

            if len(states) > 2 and player.is_trainable:
                reward = self.get_reward(state)
                # TODO: make the 8th action also terminal
                exp = TicTacToeExperience(states[-3], state, action, reward, state.is_game_over)
                player.train(exp)

            acting_side = opponent(acting_side)

        winner = state.last_acting_side if state.did_last_action_win else TicTacToeSide.NONE
        return actions, winner

    def get_reward(self, state: TicTacToeState) -> float:
        """Evaluate the given state with a reward."""
        if state.is_game_over:
            return 1.0 if state.did_last_action_win else 0.5
        return 0.0

    def select_action(self, player: TicTacToeAgent,
                      state: TicTacToeState) -> TicTacToeAction:
        """Let the player choose a valid action."""
        action = player.choose_action(state)
        while not self.env.can_apply_action(action):
            self.invalid_draws_count += 1
            if player.is_trainable:
                penalty_exp = TicTacToeExperience(state, None, action, -1.0, True)
                player.train(penalty_exp)
            action = player.choose_action(state)
        return action
