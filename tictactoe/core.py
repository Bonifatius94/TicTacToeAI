"""This module provides a game environment to play TicTacToe."""

from enum import IntEnum
from typing import List
from dataclasses import dataclass, field


class TicTacToeSide(IntEnum):
    """Representing a TicTacToe side."""
    CIRCLE = 0
    CROSS = 1
    NONE = 2

    def __repr__(self):
        if self.value == TicTacToeSide.CIRCLE:
            return 'O'
        if self.value == TicTacToeSide.CROSS:
            return 'X'
        if self.value == TicTacToeSide.NONE:
            return '_'
        raise ValueError(f'Unknown TicTacToeSide {self.value}!')


def opponent(side: TicTacToeSide) -> TicTacToeSide:
    """Get the opponent (fails for NONE)."""
    return TicTacToeSide.CROSS if side == TicTacToeSide.CIRCLE else TicTacToeSide.CIRCLE


def pos_of(row: int, col: int) -> int:
    """Get the board position index for the given row and column."""
    return row * 3 + col


@dataclass
class TicTacToeAction:
    """Representing a TicTacToe action."""
    pos: int
    token: TicTacToeSide

    def __post_init__(self):
        if self.token == TicTacToeSide.NONE:
            raise ValueError('Invalid token! Must be either CROSS or CIRCLE!')
        if self.pos not in range(9):
            raise ValueError('Invalid position! Must be within [0, 8]!')

    def __repr__(self):
        token_symbol = 'O' if self.token == TicTacToeSide.CIRCLE else 'X'
        return f'{token_symbol} -> {self.pos}'


@dataclass
class TicTacToeState:
    """Representing a TicTacToe game environment."""
    board: List[TicTacToeSide] = field(default_factory= \
        lambda: [TicTacToeSide.NONE for _ in range(9)])
    first_token: TicTacToeSide = TicTacToeSide.NONE

    @property
    def crosses(self) -> int:
        """The amount of crosses on the board."""
        return sum([1 for t in self.board if t == TicTacToeSide.CROSS])

    @property
    def circles(self) -> int:
        """The amount of circles on the board."""
        return sum([1 for t in self.board if t == TicTacToeSide.CIRCLE])

    @property
    def is_first_action(self) -> bool:
        """Determine whether it's the first action of the game."""
        return self.first_token == TicTacToeSide.NONE

    @property
    def last_acting_side(self) -> TicTacToeSide:
        """Retrieve the side that acted last."""
        first_to_draw = self.circles == self.crosses
        return opponent(self.first_token) if first_to_draw else self.first_token

    @property
    def side_to_draw(self) -> TicTacToeSide:
        """The side that has to draw next."""
        return opponent(self.last_acting_side)

    @property
    def game_outcome(self) -> TicTacToeSide:
        """Evaluate the game's outcome (only valid if game is over)."""
        if self.did_last_action_win:
            return self.last_acting_side
        return TicTacToeSide.NONE

    @property
    def is_game_over(self) -> bool:
        """Determine whether the game is over."""
        return not self.is_first_action and \
            (self.all_fields_occupied or self.did_last_action_win)

    @property
    def all_fields_occupied(self) -> bool:
        """Determine whether all fields are occupied."""
        return all(map(lambda f: f != TicTacToeSide.NONE, self.board))

    @property
    def did_last_action_win(self) -> bool:
        """Determine whether the last action scored a win."""
        if self.crosses + self.circles < 5:
            return False

        last_actor = self.last_acting_side
        all_fields_of_side = lambda l: all(map(lambda f: f == last_actor, l))

        rows = [[self.board[pos_of(row, col)] for col in range(3)] for row in range(3)]
        for row_fields in rows:
            if all_fields_of_side(row_fields):
                return True

        cols = [[self.board[pos_of(row, col)] for row in range(3)] for col in range(3)]
        for col_fields in cols:
            if all_fields_of_side(col_fields):
                return True

        diag = [self.board[pos_of(i, i)] for i in range(3)]
        diag_rev = [self.board[pos_of(2-i, i)] for i in range(3)]
        return all_fields_of_side(diag) or all_fields_of_side(diag_rev)

    def __repr__(self):
        return f'{[int(self.board[i]) for i in range(9)]}'


@dataclass
class TicTacToeExperience:
    """Representing a TicTacToe training experience."""
    state_before: TicTacToeState
    state_after: TicTacToeState
    action: TicTacToeAction
    reward: float
    is_terminal: True


def invert_state(state: TicTacToeState) -> TicTacToeState:
    """Invert the given game state."""
    if state.circles + state.crosses == 0:
        return state

    first_token = opponent(state.first_token)
    board = state.board.copy()

    for i in range(9):
        if board[i] != TicTacToeSide.NONE:
            board[i] = opponent(board[i])

    return TicTacToeState(board, first_token)


@dataclass
class TicTacToeEnv:
    """Representing a TicTacToe game environment."""
    state: TicTacToeState = TicTacToeState()

    def reset(self) -> TicTacToeState:
        """Reset the TicTacToe board."""
        self.state = TicTacToeState()
        return self.state

    def can_apply_action(self, action: TicTacToeAction) -> bool:
        """Determine whether the token can but put at the desired position."""
        return self.state.board[action.pos] == TicTacToeSide.NONE

    def apply_action(self, action: TicTacToeAction) -> TicTacToeState:
        """Apply the token to the TicTacToe board."""
        board = self.state.board.copy()
        board[action.pos] = action.token
        first_token = action.token if self.state.is_first_action else self.state.first_token
        self.state = TicTacToeState(board, first_token)
        return self.state
