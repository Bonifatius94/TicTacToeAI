import cProfile
from typing import Tuple
from tqdm import tqdm

from tictactoe.core import opponent, CIRCLE, CROSS, NONE, action_token
from tictactoe.agent import RandomTicTacToeAgent, TrainableTicTacToeAgent
from tictactoe.session import TicTacToeSession


def run_training(num_episodes: int) -> Tuple[TrainableTicTacToeAgent, TrainableTicTacToeAgent]:
    player_1 = TrainableTicTacToeAgent(side=CIRCLE)
    player_2 = TrainableTicTacToeAgent(side=CROSS)
    session = TicTacToeSession(player_1, player_2)

    print(f'training players for {num_episodes} episodes')
    for _ in tqdm(range(num_episodes)):
        session.play_game()
    print('training done!')

    return player_1, player_2


def eval_vs_random_player(num_episodes: int, player: TrainableTicTacToeAgent):
    player.expl_rate = 0.00
    rand_player = RandomTicTacToeAgent(opponent(player.side))
    session = TicTacToeSession(player, rand_player)
    wins, loses, ties = 0, 0, 0

    print('test if the players know how to win vs. random play')

    for i in range(num_episodes):
        actions, winner = session.play_game()

        if i < 10:
            print(f'game {i}, actions {actions}, winner: {winner}')

        ties += 1 if winner == NONE else 0
        wins += 1 if winner == player.side else 0
        loses += 1 if winner == opponent(player.side) else 0

    print(f'wins: {wins}, loses: {loses}, ties {ties}')


def eval_vs_good_player(num_episodes: int,
                        player_1: TrainableTicTacToeAgent,
                        player_2: TrainableTicTacToeAgent):

    session = TicTacToeSession(player_1, player_2)
    session.invalid_draws_count = 0
    player_1.expl_rate = 0.01
    player_2.expl_rate = 0.01
    wins_1st_action, wins_2nd_action, ties = 0, 0, 0

    print('test if the players know how to defend vs. a good player')

    for i in range(num_episodes):
        actions, winner = session.play_game()

        if i < 10:
            print(f'game {i}, actions {actions}, winner: {winner}')

        if winner == NONE:
            ties += 1
        elif action_token(actions[0]) == winner:
            wins_1st_action += 1
        else:
            wins_2nd_action += 1

    print(f'wins 1st action: {wins_1st_action}, wins 2nd action: {wins_2nd_action}, ties {ties}')
    print(f'players selected {session.invalid_draws_count} invalid actions')


def main():
    num_train_epochs = 50_000
    num_eval_epochs = 1_000

    player_1, player_2 = run_training(num_train_epochs)
    print('=======================================')
    print(f'player 1, weights={player_1.model.weights}, biases={player_1.model.biases}')
    print(f'player 2, weights={player_2.model.weights}, biases={player_2.model.biases}')
    print('=======================================')

    print('evaluating players')
    player_1.is_trainable = False
    player_2.is_trainable = False

    eval_vs_random_player(num_eval_epochs, player_1)
    eval_vs_random_player(num_eval_epochs, player_2)
    eval_vs_good_player(num_eval_epochs, player_1, player_2)


if __name__ == '__main__':
    # cProfile.run("main()")
    main()
