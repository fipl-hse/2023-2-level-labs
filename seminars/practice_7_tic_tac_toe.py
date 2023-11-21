"""
Programming 2022.

Seminar 5
Brainstorm from the lecture on designing a TicTacToe game
"""
# pylint:disable=too-few-public-methods

#
# class Move:
#     """
#     Store information about move: coordinates and label.
#
#     Instance attributes:
#         row (int): Index of row
#         col (int): Index of column
#         label (MarkerType): Label to put
#
#     Instance methods:
#         N/A
#     """
#     def __init__(self, index_of_row, index_of_column, label_to_put):
#         self._row = index_of_row
#         self._column = index_of_column
#         self._label = label_to_put
#
#     def git_row(self):
#         return self._row
#
#     def git_column(self):
#         return self._column
#
#     def git_label(self):
#         return self._label
#
#
# class Player:
#     """
#     Enable player functionality: store playing label (O or X), make moves.
#
#     Instance attributes:
#         label (MarkerType): label to play (O or X)
#
#     Instance methods:
#         make_move(self, row: int, col: int) -> Move: Create instance of Move
#
#     """
#     def __init__(self, label):
#         self._label = label
#
#     def get_label(self):
#         return self._label
#
#     def make_move(self, row: int, col:int):
#         players_move = (row, col, self._label)
#         return players_move
#
#
# class Game:
#     """
#     Store game status and enable moves.
#
#     Instance attributes:
#         _size (MarkerType): size of playing board (most commonly 3)
#         _board (MarkerType): playing board (most commonly 3x3)
#         _players (tuple[Player, ...]): tuple with 2 players
#         _current_player_idx (int): index of the player that should make a move next
#         _finished (MarkerType): flag if the game has finished: there was winner or tie
#
#     Instance methods:
#         _next_player(self): Update the next player to make a move.
#         _check_move(self, ...): Verify that the move can be made.
#         _register_move(self, ...): Put the move on the playing board.
#         _check_for_winner(self, ...): Check if win state is present
#         play(self, ...): Process one step of game
#     """
#     def __init__(self, size, board, players:tuple, current_player_idx: int, finished):
#         self.size = size
#         self.board = board
#         self.players = players
#         self._current_player_idx = current_player_idx
#         self._finished = finished
#
#         #self._moves_left = size ** 2 - len(self._moves)
#
#     def _next_player(self):
#
#
# class Board:
#     """
#         Store game status and enable moves.
#
#         Instance attributes:
#             _size (MarkerType): size of playing board (most commonly 3)
#             _moves_left (int): number of empty cells on the playing board
#             _moves (list[Move]): already made moves
#
#
#         Instance methods:
#             show(self, ...): Print current state of the board
#             add_move(self, ...): Add new valid move
#             get_moves(self, ...): Get already made moves
#             get_size(self, ...): Get size of board
#
#     """
#     def __int__(self, size, moves, moves_left):
#         self.size = sizegit
#         self.moves = []
#         self._moves_left = size ** 2 - len(self._moves)
#
#     def add_move(self, move):
#         self.moves.append(move)
#
#     def show(self):
#         board = []
#         for i in range(self.size):
#             row = []
#             for u in range(self.size):
#                 board.append("_")
#
# def main() -> None:
#     """
#     Launch tic-tac-toe game.
#     """
#
#     # 1. Create players
#     print('Created players')
#
#     # 2. Create game
#     print('Created game')
#
#     # 3. Make move
#     print('Made move')
#
#     # 4. Register move
#     print('Registered move')
#
#     # 5. Show current state
#     print('Showed current state')
#
#
# if __name__ == "__main__":
#     main()
