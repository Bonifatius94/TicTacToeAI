from tqdm import tqdm

from tictactoe.core import TicTacToeSide, opponent
from tictactoe.agent import RandomTicTacToeAgent, TrainableTicTacToeAgent
from tictactoe.session import TicTacToeSession


def main():
    # set training hyper-params
    num_train_epochs = 2_000_000 # training takes ~40 minutes
    num_eval_epochs = 100

    # initialize trainable players and a new game session
    player_1 = TrainableTicTacToeAgent(side=TicTacToeSide.CIRCLE)
    player_2 = TrainableTicTacToeAgent(side=TicTacToeSide.CROSS)
    session = TicTacToeSession(player_1, player_2)

    print(f'training players for {num_train_epochs} episodes')
    for i in tqdm(range(num_train_epochs)):
        session.play_game()
    print('training done!')

    print('=======================================')
    print(f'player 1, weights={player_1.model.weights}, biases={player_1.model.biases}')
    print(f'player 2, weights={player_2.model.weights}, biases={player_2.model.biases}')
    print('=======================================')

    print('evaluating players')

    player_1.is_trainable = False
    player_2.is_trainable = False
    player_1.expl_rate = 0.01
    player_2.expl_rate = 0.01

    print('test if the players know how to win vs. random play')

    wins, loses, ties = 0, 0, 0
    rand_player = RandomTicTacToeAgent(opponent(player_1.side))
    sess2 = TicTacToeSession(player_1, rand_player)

    for i in range(num_eval_epochs):
        actions, winner = sess2.play_game()

        ties += 1 if winner == TicTacToeSide.NONE else 0
        wins += 1 if winner == player_1.side else 0
        loses += 1 if winner == opponent(player_1.side) else 0

    print(f'wins: {wins}, loses: {loses}, ties {ties}')

    print('test if the players know how to defend vs. a good player')

    session.invalid_draws_count = 0
    wins_1st_action, wins_2nd_action, ties = 0, 0, 0

    for i in range(num_eval_epochs):
        actions, winner = session.play_game()

        if i < 10:
            print(f'game {i}, actions {actions}, winner: {winner}')

        if winner == TicTacToeSide.NONE:
            ties += 1
        elif actions[0].token == winner:
            wins_1st_action += 1
        else:
            wins_2nd_action += 1

    print(f'wins 1st action: {wins_1st_action}, wins 2nd action: {wins_2nd_action}, ties {ties}')
    print(f'players selected {session.invalid_draws_count} invalid actions')


if __name__ == '__main__':
    main()
