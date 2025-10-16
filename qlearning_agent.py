"""
qlearning_agent.py
A simple Q-Learning agent with a state-action table.
This agent demonstrates key RL concepts but will perform poorly due to:
1. The enormous state space of chess
2. Minimal state representation
3. Limited training time
4. Simple exploration strategy

Key concepts illustrated:
- Q-table (state-action value storage)
- Epsilon-greedy exploration
- Learning rate (alpha)
- Discount factor (gamma)
- Reward function
- State representation
"""

import random
import numpy as np
import chess
from collections import defaultdict
from agent_interface import Agent

class QLearningAgent(Agent):
    """
    A simple Q-Learning agent that learns a policy through exploration and exploitation.
    """

    def __init__(self, board: chess.Board, color: chess.Color, agent_name: str = None):
        super().__init__(board, color)
        
        # Agent identification
        self.agent_name = agent_name or f"qlearning_agent_{id(self)}"
        self.training_episodes = []  # Track training episodes

        # RL Hyperparameters - deliberately set to suboptimal values for demonstration
        self.learning_rate = 0.1 # Alpha: how much we update our Q-values
        self.discount_factor = 0.9 # Gamma: importance of future rewards
        self.epsilon = 0.3 # Exploration rate: 30% chance of random move

        # Q-table: dictionary mapping (state_hash, move) -> Q-value
        self.q_table = defaultdict(float)

        # Training state
        self.last_state = None
        self.last_move = None
        self.episode_rewards = []

        # Piece values for reward calculation (same as GreedyAgent)
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }

    def _get_state_hash(self, board: chess.Board) -> str:
        """
        Creates a simplified hash of the board state.
        This is a VERY simplistic representation - students should improve this!
        Returns: A string representation of the board's material balance and current turn.
        """
        # Count material for both sides (crude state representation)
        white_material = sum(self.piece_values.get(piece.piece_type, 0)
                             for piece in board.piece_map().values()
                             if piece.color == chess.WHITE)
        black_material = sum(self.piece_values.get(piece.piece_type, 0)
                             for piece in board.piece_map().values()
                             if piece.color == chess.BLACK)

        material_balance = white_material - black_material
        if self.color == chess.BLACK:
            material_balance = -material_balance

        # Include whose turn it is in the state
        turn_indicator = "W" if board.turn == chess.WHITE else "B"

        return f"{material_balance}_{turn_indicator}"

    def _get_reward(self, board: chess.Board, move: chess.Move = None) -> float:
        """
        Calculates the immediate reward for a move or board state.
        Students should experiment with different reward functions!
        """
        if board.is_checkmate():
            # We lost if it's our turn and we're in checkmate
            if board.turn == self.color:
                return -100.0 # Big penalty for losing
            else:
                return 100.0 # Big reward for winning

        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0 # Neutral for draws

        # Reward for captures
        if move and board.is_capture(move):
            captured_piece = board.piece_type_at(move.to_square)
            if captured_piece:
                return self.piece_values.get(captured_piece, 0)

        # Small penalty for moving to avoid stagnation
        return -0.01

    def _choose_action(self, board: chess.Board, legal_moves: list) -> chess.Move:
        """
        Epsilon-greedy action selection.
        With probability epsilon: explore (random move)
        With probability 1-epsilon: exploit (best known move)
        """
        state_hash = self._get_state_hash(board)

        # Exploration: random move
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Exploitation: choose move with highest Q-value
        best_value = float('-inf')
        best_moves = []

        for move in legal_moves:
            q_value = self.q_table.get((state_hash, move), 0.0)
            if q_value > best_value:
                best_value = q_value
                best_moves = [move]
            elif q_value == best_value:
                best_moves.append(move)

        # If we have no information, choose randomly
        if not best_moves or best_value == float('-inf'):
            return random.choice(legal_moves)

        return random.choice(best_moves)

    def _update_q_value(self, state: str, move: chess.Move, reward: float,
                        next_state: str, next_legal_moves: list):
        """
        Update the Q-value using the Q-learning formula:
        Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        current_q = self.q_table.get((state, move), 0.0)

        # Estimate of optimal future value
        if next_legal_moves:
            max_future_q = max([self.q_table.get((next_state, next_move), 0.0)
                                for next_move in next_legal_moves], default=0.0)
        else:
            max_future_q = 0.0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        self.q_table[(state, move)] = new_q

    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Main method called by the tournament driver.
        Chooses a move and updates Q-values based on the previous move.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None # No legal moves (shouldn't happen if game isn't over)

        # Get current state
        current_state = self._get_state_hash(board)

        # Choose action using epsilon-greedy policy
        chosen_move = self._choose_action(board, legal_moves)

        # Update Q-value for previous move (if we have a previous state)
        if self.last_state is not None and self.last_move is not None:
            reward = self._get_reward(board, self.last_move)
            self.episode_rewards.append(reward)

            # Update Q-value for the previous state-action pair
            self._update_q_value(self.last_state, self.last_move, reward,
                                 current_state, legal_moves)

        # Store current state and move for next update
        self.last_state = current_state
        self.last_move = chosen_move

        return chosen_move

    def reset_episode(self):
        """Reset the episode tracking variables. Call this after each game."""
        self.last_state = None
        self.last_move = None
        self.episode_rewards = []

    def get_stats(self):
        """Return some basic statistics about the agent's learning."""
        return {
            'q_table_size': len(self.q_table),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'total_episode_reward': sum(self.episode_rewards)
        }

    def save_q_table(self, filename: str = None):
        """Save the Q-table and hyperparameters to a file."""
        import pickle
        import os
        from datetime import datetime
        
        # Create agents directory if it doesn't exist
        agents_dir = "saved_agents"
        os.makedirs(agents_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agents_dir}/{self.agent_name}_{timestamp}.pkl"
        
        # Ensure filename is in agents directory
        if not filename.startswith(agents_dir):
            filename = f"{agents_dir}/{filename}"
        
        data = {
            'agent_name': self.agent_name,
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'training_episodes': len(self.training_episodes),
            'saved_at': datetime.now().isoformat(),
            'q_table_size': len(self.q_table)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent '{self.agent_name}' saved to {filename}")
        print(f"  Q-table size: {len(self.q_table)}")
        print(f"  Training episodes: {len(self.training_episodes)}")
        return filename

    def load_q_table(self, agent_name: str = None, filename: str = None):
        """Load the Q-table and hyperparameters from a file."""
        import pickle
        import os
        import glob
        
        # If no parameters provided, try to load by agent name
        if filename is None and agent_name is None:
            agent_name = self.agent_name
        
        # If agent_name provided, find the most recent file for that agent
        if filename is None and agent_name:
            agents_dir = "saved_agents"
            pattern = f"{agents_dir}/{agent_name}_*.pkl"
            files = glob.glob(pattern)
            if not files:
                print(f"No saved agent found with name '{agent_name}'")
                return False
            # Get the most recent file
            filename = max(files, key=os.path.getctime)
            print(f"Found saved agent '{agent_name}' at {filename}")
        
        # If filename provided, use it directly
        if filename:
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                
                # Update agent name if it was saved with a different name
                saved_name = data.get('agent_name', 'unknown')
                if saved_name != self.agent_name:
                    print(f"Loading agent '{saved_name}' into agent '{self.agent_name}'")
                
                self.q_table.update(data['q_table'])
                
                self.learning_rate = data.get('learning_rate', self.learning_rate)
                self.discount_factor = data.get('discount_factor', self.discount_factor)
                self.epsilon = data.get('epsilon', self.epsilon)
                
                saved_at = data.get('saved_at', 'unknown')
                q_table_size = data.get('q_table_size', len(self.q_table))
                training_episodes = data.get('training_episodes', 0)
                
                print(f"Agent loaded successfully!")
                print(f"  Q-table size: {q_table_size}")
                print(f"  Training episodes: {training_episodes}")
                print(f"  Saved at: {saved_at}")
                return True
                
            except FileNotFoundError:
                print(f"No training file found at {filename}. Starting fresh training.")
                return False
            except Exception as e:
                print(f"Error loading Q-table from {filename}: {e}")
                return False
        
        return False

class AgentManager:
    """Helper class to manage multiple saved Q-learning agents."""
    
    def __init__(self, agents_dir: str = "saved_agents"):
        self.agents_dir = agents_dir
        import os
        os.makedirs(agents_dir, exist_ok=True)
    
    def list_saved_agents(self):
        """List all saved agents with their details."""
        import glob
        import pickle
        import os
        
        pattern = f"{self.agents_dir}/*.pkl"
        files = glob.glob(pattern)
        
        if not files:
            print("No saved agents found.")
            return []
        
        agents_info = []
        for file in files:
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                
                agents_info.append({
                    'filename': file,
                    'agent_name': data.get('agent_name', 'unknown'),
                    'q_table_size': data.get('q_table_size', 0),
                    'training_episodes': data.get('training_episodes', 0),
                    'saved_at': data.get('saved_at', 'unknown'),
                    'learning_rate': data.get('learning_rate', 0),
                    'discount_factor': data.get('discount_factor', 0),
                    'epsilon': data.get('epsilon', 0)
                })
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        # Sort by saved_at timestamp (most recent first)
        agents_info.sort(key=lambda x: x['saved_at'], reverse=True)
        
        print(f"\nFound {len(agents_info)} saved agents:")
        print("-" * 80)
        for i, agent in enumerate(agents_info, 1):
            print(f"{i}. {agent['agent_name']}")
            print(f"   File: {agent['filename']}")
            print(f"   Q-table size: {agent['q_table_size']}")
            print(f"   Training episodes: {agent['training_episodes']}")
            print(f"   Saved: {agent['saved_at']}")
            print(f"   Hyperparams: α={agent['learning_rate']}, γ={agent['discount_factor']}, ε={agent['epsilon']}")
            print()
        
        return agents_info
    
    def get_agent_by_name(self, agent_name: str):
        """Get the most recent saved agent by name."""
        import glob
        import os
        
        pattern = f"{self.agents_dir}/{agent_name}_*.pkl"
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Return the most recent file
        return max(files, key=os.path.getctime)
    
    def delete_agent(self, agent_name: str):
        """Delete all saved files for a given agent name."""
        import glob
        import os
        
        pattern = f"{self.agents_dir}/{agent_name}_*.pkl"
        files = glob.glob(pattern)
        
        if not files:
            print(f"No saved agents found with name '{agent_name}'")
            return False
        
        for file in files:
            try:
                os.remove(file)
                print(f"Deleted {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
                return False
        
        print(f"Deleted {len(files)} files for agent '{agent_name}'")
        return True
    
    def create_agent_from_save(self, board, color, agent_name: str):
        """Create a new QLearningAgent and load it from a saved file."""
        agent = QLearningAgent(board, color, agent_name)
        if agent.load_q_table(agent_name):
            return agent
        else:
            print(f"Failed to load agent '{agent_name}', returning fresh agent")
            return agent

# Example training routine
def train_qlearning_agent(episodes=100, agent_name="trained_agent", save_after_training=True):
    """
    Simple training function that plays against a random agent.
    Students should expand this with better opponents and more episodes!
    
    Args:
        episodes: Number of training episodes
        agent_name: Name for the trained agent
        save_after_training: Whether to save the agent after training
    """
    from random_agent import RandomAgent

    # Create agents
    board = chess.Board()
    q_agent = QLearningAgent(board, chess.WHITE, agent_name)
    random_agent = RandomAgent(board, chess.BLACK)

    print(f"Training agent '{agent_name}' for {episodes} episodes...")

    for episode in range(episodes):
        board.reset()
        q_agent.reset_episode()
        random_agent = RandomAgent(board, chess.BLACK) # Reset opponent

        move_count = 0
        while not board.is_game_over() and move_count < 200: # Limit game length
            if board.turn == chess.WHITE:
                move = q_agent.make_move(board, 2.0)
            else:
                move = random_agent.make_move(board, 2.0)

            if move:
                board.push(move)
                move_count += 1

        # Final update after game ends
        if q_agent.last_state and q_agent.last_move:
            final_reward = q_agent._get_reward(board)
            q_agent._update_q_value(q_agent.last_state, q_agent.last_move,
                                    final_reward, "", [])

        if episode % 10 == 0:
            stats = q_agent.get_stats()
            print(f"Episode {episode}: Reward={stats['total_episode_reward']:.2f}, "
                  f"Q-table size={stats['q_table_size']}")
    
    # Save the trained agent
    if save_after_training:
        saved_file = q_agent.save_q_table()
        print(f"Training complete! Agent saved to {saved_file}")
    else:
        print("Training complete!")
    
    return q_agent

def load_trained_agent(agent_name: str, board: chess.Board, color: chess.Color):
    """
    Load a previously trained agent by name.
    
    Args:
        agent_name: Name of the saved agent
        board: Chess board instance
        color: Color for the agent
        
    Returns:
        QLearningAgent: Loaded agent or fresh agent if loading fails
    """
    agent = QLearningAgent(board, color, agent_name)
    if agent.load_q_table(agent_name):
        print(f"Successfully loaded agent '{agent_name}'")
        return agent
    else:
        print(f"Could not load agent '{agent_name}', returning fresh agent")
        return agent

if __name__ == "__main__":
    # Example usage
    print("=== Q-Learning Agent Training and Loading Demo ===")
    
    # Train a new agent
    print("\n1. Training a new agent...")
    trained_agent = train_qlearning_agent(episodes=50, agent_name="demo_agent")
    
    # List all saved agents
    print("\n2. Listing all saved agents...")
    manager = AgentManager()
    manager.list_saved_agents()
    
    # Load the trained agent
    print("\n3. Loading the trained agent...")
    board = chess.Board()
    loaded_agent = load_trained_agent("demo_agent", board, chess.WHITE)
    
    print(f"Loaded agent Q-table size: {len(loaded_agent.q_table)}")
    print("Demo complete!")