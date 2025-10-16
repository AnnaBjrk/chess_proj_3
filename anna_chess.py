import chess
import numpy as np
import torch
# set_board och make_move, vi behöver skapa en ny klass som inheritar från agent för att kunna använda metoderna.
from agent_interface import Agent


import random
import numpy as np
import chess
from collections import defaultdict
from agent_interface import Agent


class AnnaAgent(Agent):
    """
    Actor-Critic deep learning approach, class inherits from Agent class, board and color is stored in Agents
    instance variables, make_move and set_board are functions inherited from Agent.
    """

    def __init__(self, board: chess.Board, color: chess.Color, agent_name: str = None):
        super().__init__(board, color)

        # Agent identification
        self.agent_name = agent_name or f"anna_agent_{id(self)}"
        self.policy_net = ...  # instansiera modellen

    def make_move(self, board, time_limit):
        # Your brilliant RL logic goes here

        # For INFERENCE (making a move in the tournament), be safe!
        # Force the model to CPU for consistency, especially if loading
        # saved weights.
        self.model.to("cpu")

        best_move = ...
        return best_move


# vi kör med board flipping - får vi vit, behövs inte det - men får vi svart
# kör vi med board flipping, så att den kan fortsätta som vit
# så att modellen inte behöver lära sig färger.
# Det gör att tillåtna moves alltid är vita pjäser
# den ska bara spela.
def to_canonical(self, board, color):
    """Always return board from current player's perspective"""
    if color == chess.BLACK:
        # Flip board board.mirror()
        return board.transform(chess.flip_vertical)
    return board


def from_canonical(self, move, color):
    """Convert move back to actual board coordinates"""
    if color == chess.BLACK:
        # chess.square_mirror() använd denna
        return flip_move(move)
    return move


def board_to_tensor(board):
    """
    Convert chess board to tensor representation.
    Returns a 12x8x8 tensor (12 piece types, 8x8 board)
    """
    # 6 piece types * 2 colors = 12 planes
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_idx = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get rank and file (0-7)
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Determine plane index (0-5 for white, 6-11 for black)
            plane = piece_idx[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6

            tensor[plane, rank, file] = 1

    return torch.tensor(tensor)


def tensor_to_move(policy_tensor: torch.Tensor, board: chess.Board) -> chess.Move:
    """
    Convert policy tensor (73, 8, 8) to chess.Move
    Only considers moves for white pieces.

    Args:
        policy_tensor: Shape (73, 8, 8) - probabilities for each move
        board: Current board state

    Returns:
        chess.Move object with highest probability
    """
    # Directions for queen-style moves
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),    # N, NE, E, SE
        (0, -1), (-1, -1), (-1, 0), (-1, 1)  # S, SW, W, NW
    ]

    # Knight moves
    knight_moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    best_move = None
    best_prob = -999

    # Check each legal move for WHITE pieces only
    for move in board.legal_moves:
        # Get the piece at the source square
        piece = board.piece_at(move.from_square)

        # Skip if piece is None or not white
        if piece is None or piece.color != chess.WHITE:
            continue

        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

        # Find which plane this move corresponds to
        plane = None

        # Check if knight move (planes 56-63)
        if (rank_diff, file_diff) in knight_moves:
            plane = 56 + knight_moves.index((rank_diff, file_diff))

        # Check if queen-style move (planes 0-55)
        else:
            for i, (dr, df) in enumerate(directions):
                for distance in range(1, 8):
                    if rank_diff == dr * distance and file_diff == df * distance:
                        plane = i * 7 + (distance - 1)
                        break
                if plane is not None:
                    break

        # Get probability from tensor
        if plane is not None:
            prob = policy_tensor[plane, from_rank, from_file].item()
            if prob > best_prob:
                best_prob = prob
                best_move = move

    # Safety check
    if best_move is None:
        # Fallback: return first legal white move
        for move in board.legal_moves:
            piece = board.piece_at(move.from_square)
            if piece and piece.color == chess.WHITE:
                return move
        # If somehow no white moves (shouldn't happen), return any legal move
        return list(board.legal_moves)[0]

    return best_move


# Usage in make_move:
# def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
    # Your model outputs tensor of shape (73, 8, 8)
    # policy_tensor = self.model(board_state)

    # Convert tensor to chess.Move
    # move = tensor_to_move(policy_tensor, board)

    # return move


def train_model():


def main():
    run = True
    while run:
        print("Wellcome to the chess operator")
        print("What would you like to do:")
        print("1 - train a model")
        print("2 - play a game with an agent")
        print("3 - quit the program")
        choice = input("Type in you choce")
        if choice == 1:
            board = chess.Board()
            state_tensor = board_to_tensor(board)
            print(state_tensor.shape)  # torch.Size([12, 8, 8])
            pass
        elif choice == 2:
            board = chess.Board()
            pass
        elif choice == 3 or choice == q or choice == Q:
            run = False
        else:
            print("You have made an invalid choice, please try again")
        board = chess.Board()
        state_tensor = board_to_tensor(board)
        print(state_tensor.shape)  # torch.Size([12, 8, 8])
    print("programmet avslutas")
