from dataclasses import dataclass

from tictactoe.core import TicTacToeAction, TicTacToeSide
from tictactoe.session import TicTacToeSession


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
    print(actions, winner)
    assert len(actions) in [6, 8]
    assert winner == TicTacToeSide.CIRCLE


if __name__ == '__main__':
    test_session_should_handle_win()
