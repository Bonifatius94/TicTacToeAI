"""This module provides a game environment to play TicTacToe."""

from typing import List
from dataclasses import dataclass

from numba import njit, int8, int32, boolean
from numba.experimental import jitclass


NONE = 0
CIRCLE = 1
CROSS = 2


TicTacToeSide = int8
Position = int8
TicTacToeBoard = int32
TicTacToeAction = int8


def side_to_str(side: TicTacToeSide) -> str:
    if side == CIRCLE:
        return 'O'
    if side == CROSS:
        return 'X'
    if side == NONE:
        return '_'
    raise ValueError(f'Unknown TicTacToeSide {side}!')


@njit
def action_to_str(action: TicTacToeAction) -> str:
    token_str = 'O' if action_token(action) == CIRCLE else 'X'
    return f'{token_str} -> {action_pos(action)}'


@njit
def opponent(side: TicTacToeSide) -> TicTacToeSide:
    """Get the opponent."""
    assert side != NONE
    return CROSS if side == CIRCLE else CIRCLE


@njit
def pos_of(row: int8, col: int8) -> Position:
    """Get the board position index for the given row and column."""
    if row < 0 or row > 2 or col < 0 or col > 2:
        raise ValueError(f'Index out of bounds for row={row} and col={col}!')
    return row * 3 + col


@njit
def create_action(pos: Position, token: TicTacToeSide) -> TicTacToeAction:
    if token == NONE:
        raise ValueError('Invalid token! Must be either CROSS or CIRCLE!')
    if pos < 0 or pos > 8:
        raise ValueError('Invalid position! Must be within [0, 8]!')
    return (pos << 2) | token


@njit
def action_pos(action):
    return action >> 2


@njit
def action_token(action: TicTacToeAction) -> TicTacToeSide:
    return action & 0b_11


@njit
def popcnt_32(bits: int32):
    """Retrieve the amount of set bits, given a 32-bit integer."""
    # snippet source: https://stackoverflow.com/questions/407587/python-set-bits-count-popcount
    assert 0 <= bits < 0x100000000
    bits = bits - ((bits >> 1) & 0x55555555)
    bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333)
    return (((bits + (bits >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


@njit
def all_bits(bits: int32, mask: int32) -> boolean:
    """Check if the bits contain all mask bits"""
    return bits & mask == mask


@jitclass([('bits', int32)])
class TicTacToeState:
    """Representing a TicTacToe game environment."""

    def __init__(self, bits: int32=0):
        """Represent a board in lower 18 bits (2 bits each field)
        and the first_token in bits 19-20; bits 21-32 are unused"""
        self.bits = bits

    def apply_action(self, action: TicTacToeAction):
        """Put a token at the given board position."""
        if self.is_first_action:
            self.bits |= action_token(action) << 18
        self.bits |= action_token(action) << (action_pos(action) * 2)

    def field_at(self, pos: Position) -> TicTacToeSide:
        """Retrieve the field at a given position."""
        return (self.bits >> (pos * 2)) & 0b_11

    @property
    def board(self) -> List[TicTacToeSide]:
        """The board's tokens as list"""
        return [self.field_at(i) for i in range(9)]

    @property
    def first_token(self) -> TicTacToeSide:
        """The first token put on the board"""
        return self.bits >> 18

    @property
    def crosses(self) -> int8:
        """The amount of crosses on the board."""
        return popcnt_32(self.bits & 0x2AAAA)

    @property
    def circles(self) -> int8:
        """The amount of circles on the board."""
        return popcnt_32(self.bits & 0x15555)

    @property
    def is_win(self) -> boolean:
        """Determine whether the last action scored a win."""
        if self.crosses + self.circles < 5:
            return False

        is_row_win = \
            all_bits(self.bits, 0b_00_0000_0000_0001_0101) or \
            all_bits(self.bits, 0b_00_0000_0000_0010_1010) or \
            all_bits(self.bits, 0b_00_0000_0101_0100_0000) or \
            all_bits(self.bits, 0b_00_0000_1010_1000_0000) or \
            all_bits(self.bits, 0b_01_0101_0000_0000_0000) or \
            all_bits(self.bits, 0b_10_1010_0000_0000_0000)

        is_col_win = \
            all_bits(self.bits, 0b_00_0001_0000_0100_0001) or \
            all_bits(self.bits, 0b_00_0010_0000_1000_0010) or \
            all_bits(self.bits, 0b_00_0100_0001_0000_0100) or \
            all_bits(self.bits, 0b_00_1000_0010_0000_1000) or \
            all_bits(self.bits, 0b_01_0000_0100_0001_0000) or \
            all_bits(self.bits, 0b_10_0000_1000_0010_0000)

        is_diag_win = \
            all_bits(self.bits, 0b_01_0000_0001_0000_0001) or \
            all_bits(self.bits, 0b_10_0000_0010_0000_0010) or \
            all_bits(self.bits, 0b_00_0001_0001_0001_0000) or \
            all_bits(self.bits, 0b_00_0010_0010_0010_0000)

        return is_row_win or is_col_win or is_diag_win

    @property
    def is_first_action(self) -> boolean:
        """Determine whether it's the first action of the game."""
        return self.bits == 0

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
        return NONE

    @property
    def is_game_over(self) -> boolean:
        """Determine whether the game is over."""
        return not self.is_first_action and \
            (self.all_fields_occupied or self.did_last_action_win)

    @property
    def all_fields_occupied(self) -> boolean:
        """Determine whether all fields are occupied."""
        return self.crosses + self.circles == 9

    @property
    def did_last_action_win(self) -> boolean:
        """Determine whether the last action scored a win."""
        return self.crosses + self.circles >= 5 and self.is_win

    def __repr__(self) -> str:
        board = self.board
        return f'{[[board[pos_of(r, c)] for c in reversed(range(3))] for r in reversed(range(3))]}'


@dataclass
class TicTacToeExperience:
    """Representing a TicTacToe training experience."""
    state_before: TicTacToeState
    state_after: TicTacToeState
    action: TicTacToeAction
    reward: float
    is_terminal: bool


@njit
def create_bitwise_state(board: List[int8], first_token: TicTacToeSide) -> TicTacToeState:
    """Convert the common TicTacToe state to a bitwise TicTacToe state."""
    bits = sum([board[i] << (i*2) for i in range(9)])
    bits |= first_token << 18
    return TicTacToeState(bits)


@njit
def invert_state(state: TicTacToeState) -> TicTacToeState:
    """Invert the given game state."""
    if state.is_first_action:
        return state

    circles = state.bits & 0x55555
    crosses = state.bits & 0xAAAAA
    inv_bits = (circles << 1) | (crosses >> 1)
    return TicTacToeState(inv_bits)


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
        return self.state.field_at(action_pos(action)) == NONE

    def apply_action(self, action: TicTacToeAction) -> TicTacToeState:
        """Apply the token to the TicTacToe board."""
        copy = TicTacToeState(self.state.bits)
        copy.apply_action(action)
        self.state = copy
        return self.state
