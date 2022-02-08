from dataclasses import dataclass

from tictactoe.core import TicTacToeAction, TicTacToeSide, create_bitwise_state
from tictactoe.session import TicTacToeSession


def test_bitwise_board_row_win_conditions():
    state = create_bitwise_state([1, 1, 1, 2, 0, 2, 0, 0, 0], 1)
    assert state.is_game_over and int(state.game_outcome) == 1
    state = create_bitwise_state([2, 0, 2, 1, 1, 1, 0, 0, 0], 1)
    assert state.is_game_over and int(state.game_outcome) == 1
    state = create_bitwise_state([2, 0, 2, 0, 0, 0, 1, 1, 1], 1)
    assert state.is_game_over and int(state.game_outcome) == 1
    state = create_bitwise_state([2, 2, 2, 1, 0, 1, 0, 0, 0], 2)
    assert state.is_game_over and int(state.game_outcome) == 2
    state = create_bitwise_state([1, 0, 1, 2, 2, 2, 0, 0, 0], 2)
    assert state.is_game_over and int(state.game_outcome) == 2
    state = create_bitwise_state([1, 0, 1, 0, 0, 0, 2, 2, 2], 2)
    assert state.is_game_over and int(state.game_outcome) == 2


@dataclass
class MockPlayer():
    side: TicTacToeSide
    positions: list
    id: int = 0
    is_trainable: bool = False

    def choose_action(self, state):
        pos = self.positions[self.id]
        self.id += 1
        return TicTacToeAction(pos, self.side)

    def train(self, exp):
        pass


def test_session_should_handle_win():
    player_1 = MockPlayer(TicTacToeSide.CROSS, [2, 3, 0, 4, 1])
    player_2 = MockPlayer(TicTacToeSide.CIRCLE, [7, 5, 8, 6, 1])

    session = TicTacToeSession(player_1, player_2)
    actions, winner = session.play_game()

    assert len(actions) in [7, 8]
    assert winner == TicTacToeSide.CIRCLE


if __name__ == '__main__':
    test_bitwise_board_row_win_conditions()
    test_session_should_handle_win()
