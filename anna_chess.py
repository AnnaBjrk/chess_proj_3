# vi kör med board flipping - får vi vit, behövs inte det - men får vi svart
# kör vi med board flipping, så att den kan fortsätta som vit
# så att modellen inte behöver lära sig färger.
# Det gör att tillåtna moves alltid är vita pjäser
# den ska bara spela.

# metoder från Agent

# def set_board(self, board: chess.Board) -> None:
#     """
#     Update the agent's internal board state. Useful if you cache the board.
#     The tournament driver may call this to ensure the agent's board is synced.

#     Args:
#     board (chess.Board): The new, current board state.
#     """
#     self.board = board
# def __str__(self):
# """Returns a string identifier for the agent, using the class name."""
# return self.__class__.__name__
# kolla Vast AI för träning


import numpy as np
import chess
from agent_interface import Agent
import torch
import os
from datetime import datetime
from random_agent import RandomAgent
import time
import csv


class ChessEnvVsOpponent:
    """Chess environment for training against an opponent."""

    def __init__(self, opponent_agent):
        """
        Args:
            opponent_agent: An agent instance (e.g., RandomAgent)
        """
        self.board = chess.Board()
        self.opponent = opponent_agent
        self.max_moves = 200
        self.move_count = 0

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_count = 0
        observation = self._board_to_observation()
        return observation, {}

    def step(self, move):
        """
        Execute a move for the learning agent (White), then opponent moves (Black).

        Args:
            move: chess.Move object for the learning agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check if learning agent's move is legal
        if move not in self.board.legal_moves:
            return self._board_to_observation(), -1.0, True, False, {}

        # Make learning agent's move (White)
        self.board.push(move)
        self.move_count += 1

        # Check game outcome
        reward = 0.0
        terminated = False

        # Check if game ended after our move
        if self.board.is_checkmate():
            # We won!
            return self._board_to_observation(), 1.0, True, False, {}
        elif self.board.is_game_over():
            return self._board_to_observation(), 0.0, True, False, {}

        if self.move_count >= self.max_moves:
            return self._board_to_observation(), 0.0, False, True, {}

        # Opponent's turn (Black)
        try:
            opponent_move = self.opponent.make_move(self.board, time_limit=1.0)
            self.board.push(opponent_move)
            self.move_count += 1
        except Exception as e:
            # Opponent made illegal move or crashed - we win
            print(f"Opponent error: {e}")
            return self._board_to_observation(), 1.0, True, False, {}

        # Check game outcome after opponent's move
        reward = 0.0
        terminated = False

        if self.board.is_checkmate():
            # Opponent won
            reward = -1.0
            terminated = True
        elif self.board.is_game_over():
            # Draw
            reward = 0.0
            terminated = True

        truncated = self.move_count >= self.max_moves

        observation = self._board_to_observation()
        return observation, reward, terminated, truncated, {}

    def legal_moves(self):
        """Return generator of legal moves."""
        return self.board.legal_moves

    def _board_to_observation(self):
        """Convert board to 12x8x8 numpy array."""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)

        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                plane = piece_idx[piece.piece_type] + \
                    (6 if piece.color == chess.BLACK else 0)
                tensor[plane, rank, file] = 1

        return tensor


class ChessEnv:
    """Simple chess environment for RL training."""

    def __init__(self):
        self.board = chess.Board()
        self.max_moves = 200
        self.move_count = 0

    def reset(self):
        """Reset to starting position."""
        self.board = chess.Board()
        self.move_count = 0
        observation = self._board_to_observation()
        return observation, {}

    def step(self, move):
        """
        Execute a move.

        Args:
            move: chess.Move object (not an index)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Check if move is legal
        if move not in self.board.legal_moves:
            # Illegal move - end episode with penalty
            return self._board_to_observation(), -1.0, True, False, {}

        # Store whose turn it is BEFORE making the move
        was_white_turn = self.board.turn == chess.WHITE

        # Make the move
        self.board.push(move)
        self.move_count += 1

        # Check game outcome
        reward = 0.0
        terminated = False

        if self.board.is_checkmate():
            reward = 1.0
            terminated = True
        elif self.board.is_game_over():  # Stalemate, insufficient material, etc.
            reward = 0.0
            terminated = True

        truncated = self.move_count >= self.max_moves

        observation = self._board_to_observation()
        return observation, reward, terminated, truncated, {}

    def legal_moves(self):
        """Return generator of legal moves."""
        return self.board.legal_moves

    def _board_to_observation(self):
        """Convert board to 12x8x8 numpy array."""

        # If it's Black's turn, flip the board so Black sees it as "playing up"
        board_to_use = self.board.mirror() if self.board.turn == chess.BLACK else self.board
        tensor = np.zeros((12, 8, 8), dtype=np.float32)

        piece_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square in chess.SQUARES:
            piece = board_to_use.piece_at(square)
            if piece:
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                plane = piece_idx[piece.piece_type] + \
                    (6 if piece.color == chess.BLACK else 0)
                tensor[plane, rank, file] = 1

        return tensor


class AnnaAgent(Agent):
    """
    Actor-Critic deep learning approach, class inherits from Agent class, board and color is stored in Agents
    instance variables, make_move and set_board are functions inherited from Agent.
    """

    def __init__(self, board, color):
        super().__init__(board, color)

        # Agent identification
        self.agent_name = f"anna_agent_{id(self)}"
        self.policy_net = ChessModelActorCritic()  # instansiera modellen
        self.model_path = None  # "models/chosen_model.pth"

        # Load model if path is provided
        if self.model_path:
            self.policy_net.load_state_dict(
                torch.load(self.model_path, map_location='mps'))
            print(f"Loaded model from {self.model_path}")

    def make_move(self, board, time_limit):
        # Your brilliant RL logic goes here

        # For INFERENCE (making a move in the tournament), be safe!
        # Force the model to CPU for consistency, especially if loading
        # saved weights.
        """Make a move using the trained policy network."""
        self.policy_net.to(
            "mps")
        self.policy_net.eval()
        self.board = board  # Update internal board state

        # If we're black, work with a flipped board
        if self.color == chess.BLACK:
            self.board = board.mirror()
        else:
            self.board = board.copy()
        state_tensor = self.board_to_tensor()

        # Get policy from network
        with torch.no_grad():
            policy_logits, _ = self.policy_net(state_tensor)

        # Get legal moves from working board (flipped if black) using the chess library
        legal_moves = self.board.legal_moves

        # Select move - model always plays as white
        move, _ = self.select_legal_action(policy_logits, legal_moves)

        # If we're black, flip the move back to real board coordinates
        if self.color == chess.BLACK:
            from_square = chess.square_mirror(move.from_square)
            to_square = chess.square_mirror(move.to_square)
            move = chess.Move(from_square, to_square, promotion=move.promotion)

        return move

    def board_to_tensor(self):
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
            piece = self.board.piece_at(square)
            if piece:
                # Get rank and file (0-7)
                rank = chess.square_rank(square)
                file = chess.square_file(square)

                # Determine plane index (0-5 for white, 6-11 for black)
                plane = piece_idx[piece.piece_type]
                if piece.color == chess.BLACK:
                    plane += 6

                tensor[plane, rank, file] = 1
        chess_tensor_to_model = torch.tensor(tensor, dtype=torch.float32)

        return chess_tensor_to_model.unsqueeze(0)  # Add batch dimension

    def move_to_index(self, move: chess.Move) -> int:
        """
        Convert a chess.Move to a flat index in the (73, 8, 8) tensor.

    Encoding scheme:
    - Planes 0-55: Queen-style moves (8 directions × 7 distances)
                   NOTE: Queen promotions (e7-e8=Q) are encoded here as 
                   regular forward moves. The model assumes promotion to 
                   Queen is the default when a pawn reaches rank 8.
    - Planes 56-63: Knight moves (8 possible L-shapes)
    - Planes 64-72: Underpromotions (Knight/Bishop/Rook × 3 directions)
                   These are the ONLY promotions explicitly encoded.

    Args:
        move: chess.Move object

    Returns:
        Flat index (0 to 73*8*8-1)
        Args:
            move: chess.Move object
            board: Current board state (needed to check piece type)

        """
        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file

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

        # Handle underpromotions (planes 64-72)
        if move.promotion and move.promotion != chess.QUEEN:
            # Knight, Bishop, Rook promotions
            promo_map = {
                chess.KNIGHT: 0,
                chess.BISHOP: 1,
                chess.ROOK: 2
            }
            # 3 promotion types * 3 directions (left, straight, right)
            direction_offset = 0 if file_diff == - \
                1 else (1 if file_diff == 0 else 2)
            plane = 64 + promo_map[move.promotion] * 3 + direction_offset

        if plane is None:
            raise ValueError(f"Could not encode move {move}")

        # Convert to flat index: plane * 64 + rank * 8 + file
        flat_index = plane * 64 + from_rank * 8 + from_file
        return flat_index

    def index_to_move(self, flat_index: int) -> chess.Move:
        """
        Convert a flat index back to a chess.Move.

        Args:
            flat_index: Index from 0 to 73*8*8-1
            board: Current board state

        Returns:
            chess.Move object
        """
        # Unpack flat index
        plane = flat_index // 64
        square_idx = flat_index % 64
        from_rank = square_idx // 8
        from_file = square_idx % 8
        from_square = chess.square(from_file, from_rank)

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

        promotion = None

        # Knight moves (planes 56-63)
        if 56 <= plane <= 63:
            rank_diff, file_diff = knight_moves[plane - 56]

        # Underpromotions (planes 64-72)
        elif 64 <= plane <= 72:
            promo_type = (plane - 64) // 3
            direction = (plane - 64) % 3

            promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
            promotion = promo_pieces[promo_type]

            # Pawn always moves forward (rank +1), file depends on direction
            rank_diff = 1
            file_diff = -1 if direction == 0 else (0 if direction == 1 else 1)

        # Queen-style moves (planes 0-55)
        else:
            direction_idx = plane // 7
            distance = (plane % 7) + 1
            dr, df = directions[direction_idx]
            rank_diff = dr * distance
            file_diff = df * distance

        to_rank = from_rank + rank_diff
        to_file = from_file + file_diff

        # Check bounds
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Invalid move: out of bounds")

        to_square = chess.square(to_file, to_rank)

        return chess.Move(from_square, to_square, promotion=promotion)

    def select_legal_action(self, policy_logits, legal_moves):
        """
        Sample an action from legal moves only, using the policy network.

        Args:
            policy_logits: Tensor of shape (1, 73, 8, 8) or (73, 8, 8)
            legal_moves: Iterator of legal chess.Move objects
            board: Current chess.Board

        Returns:
            (chess.Move, log_prob)
        """
        # Remove batch dimension if present
        if policy_logits.dim() == 4:
            policy_logits = policy_logits.squeeze(0)

        # Flatten to (73*8*8,)
        policy_logits_flat = policy_logits.view(-1)

        # Create mask for illegal moves
        mask = torch.full_like(policy_logits_flat, float('-inf'))

        legal_moves_list = list(legal_moves)
        legal_indices = []

        # Unmask legal moves
        for move in legal_moves_list:
            try:
                idx = self.move_to_index(move)
                mask[idx] = 0
                legal_indices.append(idx)
            except ValueError as e:
                # Skip moves that can't be encoded
                print(
                    f"WARNING: Cannot encode legal move, kan vara rockad eller enpassent {move.uci()}: {e}")
                continue

        if len(legal_indices) == 0:
            raise ValueError("No legal moves could be encoded!")

        # Apply mask and get probabilities
        masked_logits = policy_logits_flat + mask
        action_probs = torch.softmax(masked_logits, dim=0)

        # Sample from distribution
        dist = torch.distributions.Categorical(action_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)

        # Convert back to move
        chosen_move = self.index_to_move(action_idx.item())

        return chosen_move, log_prob

    def train_ppo(self):
        # Single model with both heads

        # Create multiple parallel environments
        num_envs = 8  # Train on 8 games simultaneously
        envs = [ChessEnv() for _ in range(num_envs)]

        # For M1/M2/M3 Macs
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        # else:
            # device = torch.device("cpu")

        self.policy_net.to(device)
        optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)

        # Hyperparameters
        T_horizon = 2048  # Steps before update
        steps_per_env = T_horizon // num_envs  # 256 steps per env
        K_epochs = 4      # Update epochs
        gamma = 0.99
        gae_lambda = 0.95
        epsilon = 0.2

        for episode in range(10000):
            # Collect trajectory
            states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            # Reset all environments
            current_states = [env.reset()[0] for env in envs]
            for step in range(steps_per_env):
                # Stack states into batch
                state_batch = torch.FloatTensor(
                    np.array(current_states)).to(device)
                # Get actions for all environments at once
                with torch.no_grad():
                    policy_logits, value_batch = self.policy_net(state_batch)

            # Process each environment
            next_states = []
            for i, env in enumerate(envs):
                legal_moves = list(env.legal_moves())

                # If it's Black's turn, flip legal moves before passing to model
                if env.board.turn == chess.BLACK:
                    # Create flipped board for move selection
                    flipped_legal_moves = [
                        chess.Move(
                            chess.square_mirror(m.from_square),
                            chess.square_mirror(m.to_square),
                            promotion=m.promotion
                        )
                        for m in legal_moves
                    ]
                    # Model selects from flipped moves
                    action, log_prob = self.select_legal_action(
                        policy_logits[i:i+1], flipped_legal_moves
                    )
                    # Flip the chosen move back to real coordinates
                    real_move = chess.Move(
                        chess.square_mirror(action.from_square),
                        chess.square_mirror(action.to_square),
                        promotion=action.promotion
                    )
                    # Use the flipped action for encoding (what the model "thinks" it chose)
                    action_idx = self.move_to_index(action)
                else:
                    action, log_prob = self.select_legal_action(
                        policy_logits[i:i+1], legal_moves
                    )
                    real_move = action
                    action_idx = self.move_to_index(action)

                next_state, reward, terminated, truncated, _ = env.step(
                    real_move)
                done = terminated or truncated

                # Store experience
                states.append(current_states[i])
                actions.append(action_idx)
                log_probs_old.append(log_prob)
                rewards.append(reward)
                dones.append(1.0 if terminated else 0.0)
                values.append(value_batch[i].squeeze())  # ingen squeeze?

                if done:
                    next_state, _ = env.reset()

                next_states.append(next_state)

            current_states = next_states

            # *** IMPORTANT: Bootstrap value for last state ***
            # Use the last states from all environments
            with torch.no_grad():
                last_state_batch = torch.FloatTensor(
                    np.array(current_states)).to(device)
                _, last_value_batch = self.policy_net(last_state_batch)
                # Average or just use first env's value? Depends on your design
                bootstrap_value = 0 if dones[-1] == 1.0 else last_value_batch.mean().item()

            # Convert to tensors and move to device
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            log_probs_old = torch.stack(log_probs_old).detach()
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)
            values = torch.stack(values).squeeze()

            # Calculate advantages using GAE
            advantages = self.calculate_gae(
                rewards, values, dones, gamma, gae_lambda, bootstrap_value)
            returns = advantages + values.detach()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

            # PPO update for K epochs
            for _ in range(K_epochs):
                # Re-evaluate with current policy
                policy_logits, values_new = self.policy_net(states)

                # Get log probs for taken actions
                action_probs = torch.softmax(
                    policy_logits.view(len(states), -1), dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # Calculate ratio
                ratio = torch.exp(log_probs_new - log_probs_old.detach())

                # Clipped surrogate objective
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1-epsilon, 1 +
                                    epsilon) * advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = 0.5 * \
                    ((values_new.squeeze() - returns.detach()) ** 2).mean()

                # Total loss
                loss = actor_loss + critic_loss - 0.01 * entropy

                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), 0.5)
                optimizer.step()

            print(f"Episode {episode}, Return: {rewards.sum()}")

            # Save the trained model

        # Create models folder if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save model
        model_path = f'models/trained_model_{timestamp}.pth'
        torch.save(self.policy_net.state_dict(), model_path)

        print(f"Model saved to {model_path}")

    def calculate_gae(self, rewards, values, dones, gamma, gae_lambda, next_value=0):
        """
        Generalized Advantage Estimation

        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            dones: Tensor of done flags (1 if terminated, 0 otherwise)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            next_value: Bootstrap value for last state (if truncated)
        """
        advantages = []
        gae = 0

        # Detach values to prevent gradients flowing back
        values = values.detach()
        # Get the device from the input tensors
        device = rewards.device

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Use provided next_value (important for truncated episodes)
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            # TD error: δ = r + γ*V(s') - V(s)
            # If done, next_value is 0 (episode ended)
            delta = rewards[t] + gamma * \
                next_value_t * (1 - dones[t]) - values[t]

            # GAE: A = δ + γ*λ*A
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(device)

    def train_ppo_vs_opponent(self, opponent_class):
        """
        Train against a specific opponent (e.g., RandomAgent).

        Args:
            opponent_class: Class of opponent agent (e.g., RandomAgent)
        """
        # Create multiple parallel environments with opponents
        num_envs = 8
        envs = []
        for _ in range(num_envs):
            # Each environment gets its own opponent instance
            opponent = opponent_class(chess.Board(), chess.BLACK)
            envs.append(ChessEnvVsOpponent(opponent))

        # Device setup
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        # else:
            # device = torch.device("cpu")

        self.policy_net.to(device)
        optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=1e-4)  # lr=3e-4

        # Hyperparameters
        T_horizon = 2048
        steps_per_env = T_horizon // num_envs
        K_epochs = 3  # 4
        gamma = 0.99
        gae_lambda = 0.95
        epsilon = 0.1  # 0.2

        # Early stopping tracking
        best_win_rate = 0.0
        best_episode = 0
        episodes_without_improvement = 0
        patience = 200  # Stop if no improvement for 300 episodes

        # ========== ADD CSV SETUP ==========

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'training_stats_{timestamp}.csv'

        # Create CSV with header
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Win_Rate', 'Wins',
                            'Losses', 'Avg_Reward', 'ETA_Hours'])

        print(f"Stats saving to: {csv_filename}\n")
        # ================================

        # ========== TRACKING ==========
        start_time = time.time()
        episode_wins = []
        episode_losses = []
        episode_rewards = []

        print("\n" + "="*70)
        print("TRAINING vs RandomAgent")
        print("="*70)

        for episode in range(1500):  # 10000
            states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

            # Reset all environments
            current_states = [env.reset()[0] for env in envs]

            for step in range(steps_per_env):
                # Stack states into batch
                state_batch = torch.FloatTensor(
                    np.array(current_states)).to(device)

                # Get actions for all environments at once
                with torch.no_grad():
                    policy_logits, value_batch = self.policy_net(state_batch)

                # Process each environment
                next_states = []
                for i, env in enumerate(envs):
                    legal_moves = list(env.legal_moves())

                    # Select action for this environment
                    action, log_prob = self.select_legal_action(
                        policy_logits[i:i+1], legal_moves
                    )
                    action_idx = self.move_to_index(action)

                    next_state, reward, terminated, truncated, _ = env.step(
                        action)
                    done = terminated or truncated

                    # Store experience
                    states.append(current_states[i])
                    actions.append(action_idx)
                    log_probs_old.append(log_prob)
                    rewards.append(reward)
                    dones.append(1.0 if terminated else 0.0)
                    values.append(value_batch[i].squeeze())

                    if done:
                        next_state, _ = env.reset()

                    next_states.append(next_state)

                current_states = next_states

            # Bootstrap value calculation
            with torch.no_grad():
                last_state_batch = torch.FloatTensor(
                    np.array(current_states)).to(device)
                _, last_value_batch = self.policy_net(last_state_batch)
                bootstrap_value = 0 if dones[-1] == 1.0 else last_value_batch[0].item()

            # Convert to tensors and move to device
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).to(device)
            log_probs_old = torch.stack(log_probs_old).detach()
            rewards = torch.FloatTensor(rewards).to(device)
            dones = torch.FloatTensor(dones).to(device)
            values = torch.stack(values)

            # Calculate advantages using GAE
            advantages = self.calculate_gae(
                rewards, values, dones, gamma, gae_lambda, bootstrap_value)
            returns = advantages + values.detach()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

            # PPO update for K epochs
            for _ in range(K_epochs):
                policy_logits, values_new = self.policy_net(states)

                action_probs = torch.softmax(
                    policy_logits.view(len(states), -1), dim=-1)
                dist = torch.distributions.Categorical(action_probs)
                log_probs_new = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs_new - log_probs_old.detach())

                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1-epsilon, 1 +
                                    epsilon) * advantages.detach()
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * \
                    ((values_new.squeeze() - returns.detach()) ** 2).mean()

                loss = actor_loss + critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), 0.5)
                optimizer.step()

            # Track wins/losses
            total_reward = rewards.sum().item()
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

            # ========== TRACK STATISTICS ==========
            total_reward = rewards.sum().item()
            wins = (rewards == 1.0).sum().item()
            losses = (rewards == -1.0).sum().item()

            episode_wins.append(wins)
            episode_losses.append(losses)
            episode_rewards.append(total_reward)

            # ========== PRINT EVERY 50 EPISODES ==========
            if episode % 50 == 0:  # 50
                # Last 50 episodes stats
                recent_wins = sum(episode_wins[-50:])
                recent_losses = sum(episode_losses[-50:])
                recent_rewards = episode_rewards[-50:] if len(
                    episode_rewards) >= 50 else episode_rewards

                total_games = recent_wins + recent_losses
                win_rate = (recent_wins / total_games *
                            100) if total_games > 0 else 0
                avg_reward = np.mean(recent_rewards)

                # Time estimate
                elapsed = time.time() - start_time
                eta_hours = (elapsed / (episode + 1)) * \
                    (10000 - episode) / 3600

                print(f"Ep {episode:4d} | Win Rate: {win_rate:5.1f}% | "
                      f"W: {recent_wins:3d} L: {recent_losses:3d} | "
                      f"Avg Reward: {avg_reward:+.2f} | ETA: {eta_hours:.1f}h")

                # ========== CHECK FOR BEST MODEL ==========
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_episode = episode
                    episodes_without_improvement = 0

                    # Save best model
                    best_model_path = f'models/best_model_{timestamp}.pth'
                    torch.save(self.policy_net.state_dict(), best_model_path)
                    print(f"Ep {episode:4d} | Win Rate: {win_rate:5.1f}% | "
                          f"W: {recent_wins:3d} L: {recent_losses:3d} | "
                          f"Avg Reward: {avg_reward:+.2f} | ETA: {eta_hours:.1f}h ⭐ NEW BEST!")
                else:
                    episodes_without_improvement += 50
                    print(f"Ep {episode:4d} | Win Rate: {win_rate:5.1f}% | "
                          f"W: {recent_wins:3d} L: {recent_losses:3d} | "
                          f"Avg Reward: {avg_reward:+.2f} | ETA: {eta_hours:.1f}h "
                          f"(Best: {best_win_rate:.1f}% @ep{best_episode})")

                # ========== ADD: SAVE TO CSV ==========
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, win_rate, recent_wins, recent_losses,
                                     avg_reward, eta_hours])
                # ====================================

                # Save checkpoint every 500
                if episode % 250 == 0 and episode > 0:
                    checkpoint_path = f'models/checkpoint_ep{episode}.pth'
                    torch.save(self.policy_net.state_dict(), checkpoint_path)
                    print(f"  → Saved checkpoint: {checkpoint_path}")

                # ========== EARLY STOPPING CHECK ==========
                if episodes_without_improvement >= patience:
                    print("\n" + "="*70)
                    print(f"⚠️  EARLY STOPPING TRIGGERED!")
                    print(f"No improvement for {patience} episodes.")
                    print(
                        f"Best win rate: {best_win_rate:.1f}% at episode {best_episode}")
                    print(
                        f"Best model saved at: models/best_model_{timestamp}.pth")
                    print("="*70)
                    break

         # ========== FINAL SUMMARY ==========
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")

        # Check if we have enough episodes for final stats
        if len(episode_wins) >= 100:
            final_wins = sum(episode_wins[-100:])
            final_losses = sum(episode_losses[-100:])
            final_wr = (final_wins / (final_wins + final_losses) *
                        100) if (final_wins + final_losses) > 0 else 0
            print(f"Final Win Rate (last 100 eps): {final_wr:.1f}%")

        print(f"Best Win Rate: {best_win_rate:.1f}% at episode {best_episode}")
        print(f"Total Training Time: {total_time/3600:.2f} hours")
        print(f"Training stats saved to: {csv_filename}")
        print(f"Best model saved to: models/best_model_{timestamp}.pth")
        print("="*70 + "\n")

        # Save model
        os.makedirs('models', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'models/trained_vs_random_final{timestamp}.pth'
        torch.save(self.policy_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")


class ChessModelActorCritic(torch.nn.Module):
    def __init__(self):
        super(ChessModelActorCritic, self).__init__()
        # Define the neural network layers here
        self.conv1 = torch.nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 512)

        # Actor head
        self.actor_fc = torch.nn.Linear(
            512, 73 * 8 * 8)  # 73 possible move types

        # Critic head
        self.critic_fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))

        # Actor output
        policy_logits = self.actor_fc(x)
        # Reshape to (batch_size, 73, 8, 8)
        policy_logits = policy_logits.view(-1, 73, 8, 8)

        # Critic output
        value = self.critic_fc(x)

        return policy_logits, value


def main():
    run = True

    while run:
        print("Welcome to the chess operator")
        print("What would you like to do:")
        print("1 - train a model")
        print("2 - play a game against random agent")
        print("3 - ladda en sparad modell och spela mot random agent (kommer senare)")
        print("4 - quit the program")
        choice = input("Type in your choice:  ")
        if choice == "1":
            annasagent = AnnaAgent(chess.Board(), chess.WHITE)
            annasagent.train_ppo()
        elif choice == "2":
            annasagent = AnnaAgent(chess.Board(), chess.WHITE)
            annasagent.train_ppo_vs_opponent(RandomAgent)

        elif choice == "3":
            annasagent = AnnaAgent(chess.Board(), chess.WHITE)
            # Load checkpoint
            print("\nAvailable checkpoints:")

            if os.path.exists('models'):
                files = [f for f in os.listdir('models') if f.endswith('.pth')]
                for i, f in enumerate(files):
                    print(f"  {i+1}. {f}")

                file_choice = input("Enter number to load: ")
                try:
                    file_idx = int(file_choice) - 1
                    file_path = f'models/{files[file_idx]}'
                    annasagent.policy_net.load_state_dict(
                        torch.load(file_path, map_location='mps')
                    )
                    print(f"✓ Loaded model from {file_path}")
                except:
                    print("Invalid choice")
            else:
                print("No models folder found!")
        annasagent.train_ppo_vs_opponent(RandomAgent)

        elif choice == "4" or choice == "q" or choice == "Q":
            run = False
        else:
            print("You have made an invalid choice, please try again")

    print("programmet avslutas")


if __name__ == "__main__":
    main()
