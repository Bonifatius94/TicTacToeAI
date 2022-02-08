"""This module provides a game environment to play TicTacToe."""

from enum import IntEnum
from typing import List, Protocol
from dataclasses import dataclass, field


class TicTacToeSide(IntEnum):
    """Representing a TicTacToe side."""
    NONE = 0
    CIRCLE = 1
    CROSS = 2

    def __repr__(self):
        if self.value == TicTacToeSide.CIRCLE:
            return 'O'
        if self.value == TicTacToeSide.CROSS:
            return 'X'
        if self.value == TicTacToeSide.NONE:
            return '_'
        raise ValueError(f'Unknown TicTacToeSide {self.value}!')

    def __str__(self):
        return self.__repr__()


def opponent(side: TicTacToeSide) -> TicTacToeSide:
    """Get the opponent."""
    assert side != TicTacToeSide.NONE
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


class TicTacToeState(Protocol):
    """Representing a TicTacToe game environment."""
    board: List[TicTacToeSide]
    first_token: TicTacToeSide
    crosses: int
    circles: int
    is_first_action: bool
    last_acting_side: TicTacToeSide
    side_to_draw: TicTacToeSide
    game_outcome: TicTacToeSide
    is_game_over: bool
    all_fields_occupied: bool
    did_last_action_win: bool


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


def popcnt_32(bits: int):
    """Retrieve the amount of set bits, given a 32-bit integer."""
    # snippet source: https://stackoverflow.com/questions/407587/python-set-bits-count-popcount
    assert 0 <= bits < 0x100000000
    bits = bits - ((bits >> 1) & 0x55555555)
    bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333)
    return (((bits + (bits >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


def all_bits(bits: int, mask: int) -> bool:
    """Check if the bits contain all mask bits"""
    return bits & mask == mask


@dataclass
class BitwiseTicTacToeState:
    """Representing a TicTacToe game environment."""
    # store the board in lower 18 bits (2 bits each field)
    # and the first_token in bits 19-20; bits 21-32 are unused
    bits: int = 0

    @property
    def board(self) -> List[TicTacToeSide]:
        """The board's tokens as list"""
        print(self.bits)
        return [TicTacToeSide((self.bits >> (i*2)) & 3) for i in range(9)]

    @property
    def first_token(self) -> TicTacToeSide:
        """The first token put on the board"""
        return TicTacToeSide(self.bits >> 18)

    @property
    def crosses(self) -> int:
        """The amount of crosses on the board."""
        return popcnt_32(self.bits & 0x0002AAAA)

    @property
    def circles(self) -> int:
        """The amount of circles on the board."""
        return popcnt_32(self.bits & 0x00015555)

    @property
    def is_first_action(self) -> bool:
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
        return TicTacToeSide.NONE

    @property
    def is_game_over(self) -> bool:
        """Determine whether the game is over."""
        return not self.is_first_action and \
            (self.all_fields_occupied or self.did_last_action_win)

    @property
    def all_fields_occupied(self) -> bool:
        """Determine whether all fields are occupied."""
        return self.crosses + self.circles == 9

    @property
    def did_last_action_win(self) -> bool:
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


def create_bitwise_state(board: List[TicTacToeSide],
                         first_token: TicTacToeSide) -> BitwiseTicTacToeState:
    """Convert the common TicTacToe state to a bitwise TicTacToe state."""
    bits = sum([int(board[i]) << (i*2) for i in range(9)])
    bits |= int(first_token) << 18
    return BitwiseTicTacToeState(bits)


def create_bitwise_state(board: List[int],
                         first_token: int) -> BitwiseTicTacToeState:
    """Convert the common TicTacToe state to a bitwise TicTacToe state."""
    bits = sum([board[i] << (i*2) for i in range(9)])
    bits |= first_token << 18

    try:
        _ = BitwiseTicTacToeState(bits).bits
    except:
        print(f'{bits:b}')

    return BitwiseTicTacToeState(bits)


def invert_state(state: TicTacToeState) -> TicTacToeState:
    """Invert the given game state."""
    if state.is_first_action:
        return state

    # optimized procedure for bitwise game state
    if isinstance(state, BitwiseTicTacToeState):
        state: BitwiseTicTacToeState
        shift = 1 if state.first_token == TicTacToeSide.CIRCLE else -1

        try:
            bits = state.bits << shift
            _ = BitwiseTicTacToeState(bits).bits
        except:
            print(f'{bits:b}')

        return BitwiseTicTacToeState(state.bits << shift)

    # default to fallback case
    first_token = opponent(state.first_token)
    board = state.board.copy()

    for i in range(9):
        if board[i] != TicTacToeSide.NONE:
            board[i] = opponent(board[i])

    return create_bitwise_state(board, first_token)


@dataclass
class TicTacToeEnv:
    """Representing a TicTacToe game environment."""
    state: TicTacToeState = BitwiseTicTacToeState()

    def reset(self) -> TicTacToeState:
        """Reset the TicTacToe board."""
        self.state = BitwiseTicTacToeState()
        return self.state

    def can_apply_action(self, action: TicTacToeAction) -> bool:
        """Determine whether the token can but put at the desired position."""

        # optimized procedure for bitwise game state
        if isinstance(self.state, BitwiseTicTacToeState):
            return not (self.state.bits << (action.pos * 2)) & 3

        # default to fallback case
        return self.state.board[action.pos] == TicTacToeSide.NONE

    def apply_action(self, action: TicTacToeAction) -> TicTacToeState:
        """Apply the token to the TicTacToe board."""

        # optimized procedure for bitwise game state
        if isinstance(self.state, BitwiseTicTacToeState):
            bits = self.state.bits
            bits |= int(action.token) << (action.pos * 2)
            first_token = action.token if self.state.is_first_action else self.state.first_token
            bits |= int(first_token) << 18
            self.state = BitwiseTicTacToeState(bits)
            return self.state

        # default to fallback case
        board = self.state.board.copy()
        board[action.pos] = action.token
        first_token = action.token if self.state.is_first_action else self.state.first_token
        self.state = create_bitwise_state(board, first_token)
        return self.state


# @dataclass
# class SimpleTicTacToeState:
#     """Representing a TicTacToe game environment."""
#     board: List[TicTacToeSide] = field(default_factory= \
#         lambda: [TicTacToeSide.NONE for _ in range(9)])
#     first_token: TicTacToeSide = TicTacToeSide.NONE

#     @property
#     def crosses(self) -> int:
#         """The amount of crosses on the board."""
#         return sum([1 for t in self.board if t == TicTacToeSide.CROSS])

#     @property
#     def circles(self) -> int:
#         """The amount of circles on the board."""
#         return sum([1 for t in self.board if t == TicTacToeSide.CIRCLE])

#     @property
#     def is_first_action(self) -> bool:
#         """Determine whether it's the first action of the game."""
#         return self.first_token == TicTacToeSide.NONE

#     @property
#     def last_acting_side(self) -> TicTacToeSide:
#         """Retrieve the side that acted last."""
#         first_to_draw = self.circles == self.crosses
#         return opponent(self.first_token) if first_to_draw else self.first_token

#     @property
#     def side_to_draw(self) -> TicTacToeSide:
#         """The side that has to draw next."""
#         return opponent(self.last_acting_side)

#     @property
#     def game_outcome(self) -> TicTacToeSide:
#         """Evaluate the game's outcome (only valid if game is over)."""
#         if self.did_last_action_win:
#             return self.last_acting_side
#         return TicTacToeSide.NONE

#     @property
#     def is_game_over(self) -> bool:
#         """Determine whether the game is over."""
#         return not self.is_first_action and \
#             (self.all_fields_occupied or self.did_last_action_win)

#     @property
#     def all_fields_occupied(self) -> bool:
#         """Determine whether all fields are occupied."""
#         return all(map(lambda f: f != TicTacToeSide.NONE, self.board))

#     @property
#     def did_last_action_win(self) -> bool:
#         """Determine whether the last action scored a win."""
#         if self.crosses + self.circles < 5:
#             return False

#         last_actor = self.last_acting_side
#         all_fields_of_side = lambda l: all(map(lambda f: f == last_actor, l))

#         rows = [[self.board[pos_of(row, col)] for col in range(3)] for row in range(3)]
#         for row_fields in rows:
#             if all_fields_of_side(row_fields):
#                 return True

#         cols = [[self.board[pos_of(row, col)] for row in range(3)] for col in range(3)]
#         for col_fields in cols:
#             if all_fields_of_side(col_fields):
#                 return True

#         diag = [self.board[pos_of(i, i)] for i in range(3)]
#         diag_rev = [self.board[pos_of(2-i, i)] for i in range(3)]
#         return all_fields_of_side(diag) or all_fields_of_side(diag_rev)

#     def __repr__(self):
#         return f'{[int(self.board[i]) for i in range(9)]}'
