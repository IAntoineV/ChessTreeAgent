import torch
import chess
import chess.engine
import numpy as np
from copy import deepcopy
from collections import defaultdict

# ===== MOCKS =====

TOTAL_MOVES = 10  # small for demo
legal_moves = [
    chess.Move.from_uci(m)
    for m in ["e2e4","d2d4","g1f3","c2c4","f2f4","b1c3","e2e3","d2d3","g2g3","b2b3"]
]
policy_index = {i: m for i, m in enumerate(legal_moves)}
policy_inverse_map = {m: i for i, m in policy_index.items()}

def board_encoding(board):
    return torch.randn(1, 4)

class MockPolicy:
    def __init__(self, logits):
        self.probs = torch.softmax(logits, dim=0)

class MockEncoder:
    def __call__(self, x):
        bs = x.size(0)
        emb = torch.randn(bs, 4)
        pol = [MockPolicy(torch.randn(TOTAL_MOVES)) for _ in range(bs)]
        val = torch.tanh(torch.randn(bs))
        return emb, pol, val

# ===== TREE STRUCTURE =====

class TreeNode:
    def __init__(self, board, move=None, parent=None, value=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.value = value
        self.children = []
        self.played_moves = set()
        self.is_terminal = board.is_game_over()
        self.root_player = parent.root_player if parent else board.turn

    def add_child(self, move_id):
        """Apply move by index and return new child node."""
        mv = policy_index[move_id]
        new_bd = deepcopy(self.board)
        new_bd.push(mv)
        self.played_moves.add(move_id)
        child = TreeNode(new_bd, move=mv, parent=self)
        self.children.append(child)
        return child

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children


# ===== STOCKFISH EVALUATION =====

def evaluate_with_stockfish(board: chess.Board,
                            engine: chess.engine.SimpleEngine,
                            root_player: bool,
                            depth: int = 12) -> float:
    """
    Apply Stockfish to `board` to the given depth.
    Returns a pawn-unit score from root_player's perspective.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=100000) or 0
    if root_player == chess.BLACK:
        score = -score
    return score / 100.0

# ===== ENVIRONMENT =====

class BatchedChessTreeEnv:
    """
    Actions:
      0: reset to root
      1: sample a move from policy (normal)
      2: sample without repeating moves at this node
      3: backtrack one move
    """

    def __init__(self, encoder, board_sampler, max_steps=10, num_envs=2, stockfish_path="stockfish"):
        self.encoder = encoder
        self.board_sampler = board_sampler
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def reset(self):
        self.roots = [TreeNode(next(self.board_sampler)) for _ in range(self.num_envs)]
        self.current = list(self.roots)
        self.step_count = 0

        boards = [n.board for n in self.current]
        x = torch.cat([board_encoding(b) for b in boards], dim=0)
        emb, pol, val = self.encoder(x)
        self.state = emb.unsqueeze(1)
        self.policies, self.values = pol, val

        for i, n in enumerate(self.current):
            n.value = self.values[i].item()

        return self.state

    def step(self, actions):
        for i, a in enumerate(actions.tolist()):
            node = self.current[i]

            if a == 0:
                # Action 0: reset to root
                self.current[i] = self.roots[i]

            elif a == 1:
                # Action 1: sample move normally
                idx = torch.multinomial(self.policies[i].probs, 1).item()
                self.current[i] = node.add_child(idx)

            elif a == 2:
                # Action 2: sample without repeating moves at this node
                probs = self.policies[i].probs.clone()
                mask = torch.ones_like(probs, dtype=torch.bool)
                for used in node.played_moves:
                    mask[used] = False
                masked = probs * mask
                if masked.sum() <= 0:
                    node.played_moves.clear()
                    masked = probs
                masked = masked / masked.sum()
                idx = torch.multinomial(masked, 1).item()
                self.current[i] = node.add_child(idx)

            elif a == 3:
                # Action 3: backtrack one move
                parent = node.get_parent()
                if parent:
                    self.current[i] = parent

        self.step_count += 1

        boards = [n.board for n in self.current]
        x = torch.cat([board_encoding(b) for b in boards], dim=0)
        emb, pol, val = self.encoder(x)
        self.state = torch.cat([self.state, emb.unsqueeze(1)], dim=1)
        self.policies, self.values = pol, val

        for i, n in enumerate(self.current):
            n.value = self.values[i].item()

        dones = np.array([self.step_count >= self.max_steps] * self.num_envs)
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        infos = [{} for _ in range(self.num_envs)]
        return self.state, rewards, dones, infos

    def close(self):
        self.engine.quit()

# ===== EXAMPLE =====

if __name__ == "__main__":
    STOCKFISH_PATH = ... # include your stockfish path here
    def board_sampler():
        while True:
            yield chess.Board()

    env = BatchedChessTreeEnv(MockEncoder(), board_sampler(), max_steps=3, num_envs=2, stockfish_path=STOCKFISH_PATH)
    state = env.reset()
    print("Initial state:", state.shape)
    for t in range(3):
        actions = torch.tensor([1, 2])
        state, rewards, dones, _ = env.step(actions)
        print(f"Step {t+1}: rewards={rewards}, dones={dones}")
    env.close()
