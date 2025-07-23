import torch
import chess
import numpy as np
from copy import deepcopy
from collections import defaultdict
from .utils import board_encoding, policy_index, policy_inverse_map

class BatchedChessTreeEnv:
    """
    Batched environment exploring a move-tree in parallel.
    Actions:
      0: reset to root
      1: sample a move from policy head (normal sampling)
      2: sample from policy head without reusing moves at each position
      3: backtrack one move
    Episodes fixed-length, never truncated early.
    """
    def __init__(self, encoder, board_sampler, max_steps=50, num_envs=4):
        super().__init__()
        self.num_envs       = num_envs
        self.encoder        = encoder
        self.board_sampler  = board_sampler
        self.max_steps      = max_steps
        self.current_step   = 0

    def reset(self):
        # initialize root boards per env
        self.roots = [next(self.board_sampler) for _ in range(self.num_envs)]
        # board histories for backtrack logic
        self.board_histories = [[deepcopy(r)] for r in self.roots]
        # tried moves per fen per env for no-reuse sampling
        self.tried_moves = [defaultdict(set) for _ in range(self.num_envs)]
        self.current_step = 0

        # initial encoding
        boards = [h[-1] for h in self.board_histories]
        pre_state = torch.cat([board_encoding(b) for b in boards], dim=0)
        emb, pol, val = self.encoder(pre_state)
        self.state = emb.unsqueeze(1)    # shape: (num_envs,1,emb_dim)
        self.policies = pol
        self.values = val

        return self.state

    def step(self, actions):
        """Step through each env with batched actions."""
        for i, a in enumerate(actions.tolist()):
            history = self.board_histories[i]
            tried   = self.tried_moves[i]
            board   = history[-1]
            fen     = board.fen()

            if a == 0:
                # reset to root
                history[:] = [deepcopy(self.roots[i])]
                tried.clear()

            elif a == 1:
                # normal policy sampling
                policy = self.policies[i]
                probs = policy.probs.cpu().numpy()
                idx = np.random.choice(len(probs), p=probs)
                move = policy_index[idx]
                new_bd = deepcopy(board)
                new_bd.push(move)
                history.append(new_bd)

            elif a == 2:
                # policy sampling without reuse
                policy = self.policies[i]
                idx = self._sample_from_policy(policy, tried[fen])
                move = policy_index[idx]
                new_bd = deepcopy(board)
                new_bd.push(move)
                history.append(new_bd)

            elif a == 3:
                # backtrack one
                if len(history) > 1:
                    history.pop()
                tried[fen].clear()

        self.current_step += 1

        # encode new positions
        boards = [h[-1] for h in self.board_histories]
        pre_state = torch.cat([board_encoding(b) for b in boards], dim=0)
        emb, pol, val = self.encoder(pre_state)
        self.state = torch.cat([self.state, emb.unsqueeze(1)], dim=1)
        self.policies = pol
        self.values = val


        # compute rewards and dones
        rewards = self.values.squeeze().detach().cpu().numpy().astype(np.float32)
        dones = np.array([self.current_step >= self.max_steps] * self.num_envs)
        infos = [{} for _ in range(self.num_envs)]

        return self.state, rewards, dones, infos

    def _sample_from_policy(self, policy, exclude_idxs):
        """Sample according to policy distribution, excluding indexes in exclude_idxs."""
        probs = policy.probs.cpu().numpy().copy()
        mask = np.ones_like(probs)
        mask[list(exclude_idxs)] = 0.0
        probs *= mask
        total = probs.sum()
        if total <= 0:
            # reset exclusion if all moves used
            exclude_idxs.clear()
            probs = policy.probs.cpu().numpy().copy()
            total = probs.sum()
        probs /= total
        choice = np.random.choice(len(probs), p=probs)
        exclude_idxs.add(choice)
        return choice

    def close(self):
        pass