# -*- coding: utf-8 -*-
"""
dqn_single_two_cell.py
----------------------
Q-Learning vs DQN for single-cell and two-cell D2D-underlay cellular systems.

Usage (Python 3.10+ recommended):
  py -3 dqn_single_two_cell.py          # Windows
  python3 dqn_single_two_cell.py        # Linux/Mac

Dependencies:
  numpy, matplotlib
  torch (optional but required for DQN; Q-learning always runs)

Outputs:
  single_avg.png, single_best.png
  twocell_avg.png, twocell_best.png
"""

import math
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("PyTorch not available; DQN will be skipped. Install CPU torch to enable DQN.")

from collections import deque, defaultdict

# =====================================================
# Utilities
# =====================================================
def dbm_to_w(dbm: float) -> float:
    # P[W] = 10^((dBm - 30)/10)
    return 10 ** ((dbm - 30.0) / 10.0)

def db_to_lin(db: float) -> float:
    return 10 ** (db / 10.0)

# =====================================================
# Base Single-Cell Environment
# =====================================================
class SingleCellEnv:
    """
    Single-cell D2D-underlay cellular environment (uplink period).

    Implements the capacity and SINR structure aligned with the user's paper:
      - Eq. (1): SINR for CUE_i at gNB (uplink) with D2D interference
      - Eq. (2): SINR for DRUE_j with CUE_m and co-channel D2D interference
      - Eq. (3): Total capacity is sum of CUE and DUE capacities over B*log2(1+SINR)
      - Eq. (4): Average/required CUE Tx power meeting SINR threshold (capped by Pt_max)
      - Eq. (5): (not explicitly maximized here; reward is total capacity; EE can be computed if needed)

    State vector is a smoothed/normalized feature of current assignment & geometry.
    Action space is selecting (D2D index, channel index) pair.
    """
    def __init__(self,
                 M=8, N=3,     # D2D pairs, CUEs (also # of subchannels = N)
                 R=600.0, L=20.0,
                 Pt_d2d=0.01, Pt_cue_max=2.0,
                 B_MHz=0.15, alpha=4.0, N0_dbm=-90.0, SINRTh_db=4.6,
                 gain_pl=2.5e5,
                 state_ema=0.8, reward_ema=0.9,
                 move_sigma_ratio=0.02,
                 seed=2025):
        self.M, self.N = int(M), int(N)
        self.R, self.L = float(R), float(L)
        self.Pt_d2d = float(Pt_d2d)
        self.Pt_cue_max = float(Pt_cue_max)
        self.B = float(B_MHz)  # MHz, use as "B" in Mbps throughput (B * log2(1+SINR))
        self.alpha = float(alpha)
        self.N0 = dbm_to_w(N0_dbm)
        self.SINRTh = db_to_lin(SINRTh_db)
        self.gain_PL = float(gain_pl)
        self.state_ema = float(state_ema)
        self.reward_ema = float(reward_ema)
        self.move_sigma = self.R * float(move_sigma_ratio)
        self.rng = np.random.default_rng(seed)

        self._layout_ready = False
        self.prev_feature = None
        self.reward_smooth = 0.0
        self.episode_counter = 0

    # ------------- Layout Helpers -------------
    def _uniform_disc(self, n, radius):
        r = np.sqrt(self.rng.random(n)) * radius
        th = self.rng.random(n) * 2.0 * np.pi
        return np.stack([r*np.cos(th), r*np.sin(th)], axis=1)

    def _place_nodes(self):
        # N CUEs (also N subchannels); M D2D pairs (DTUE/DRUE)
        self.CUE = self._uniform_disc(self.N, self.R)
        self.DTUE = self._uniform_disc(self.M, self.R)
        # place DRUE near DTUE within radius L
        self.DRUE = self.DTUE + self._uniform_disc(self.M, self.L)
        # precompute distances to gNB at origin
        self.r_CUE = np.linalg.norm(self.CUE, axis=1)              # |CUE_i - gNB|
        self.r_DTUE = np.linalg.norm(self.DTUE, axis=1)            # |DTUE_k - gNB|
        self.r_Dpair = np.linalg.norm(self.DTUE - self.DRUE, axis=1)  # |DTUE_j - DRUE_j|
        self._layout_ready = True

    # ------------- RL API -------------
    def reset(self):
        if not self._layout_ready:
            self._place_nodes()
        else:
            # partial mobility: move one user slightly
            if self.rng.random() < 0.4:
                # move a random CUE
                i = self.rng.integers(0, self.N)
                self.CUE[i] += self.rng.normal(0, self.move_sigma, 2)
                self.r_CUE[i] = np.linalg.norm(self.CUE[i])
            else:
                # move a random D2D pair
                j = self.rng.integers(0, self.M)
                self.DTUE[j] += self.rng.normal(0, self.move_sigma, 2)
                self.DRUE[j]  = self.DTUE[j] + self._uniform_disc(1, self.L*0.25)[0]
                self.r_DTUE[j] = np.linalg.norm(self.DTUE[j])
                self.r_Dpair[j] = np.linalg.norm(self.DTUE[j] - self.DRUE[j])

        # random initial channel reuse mapping for D2D pairs in [0..N-1]
        self.assign = self.rng.integers(0, self.N, size=self.M)
        self.prev_feature = None
        self.reward_smooth = 0.0
        self.episode_counter += 1
        return self._feature()

    @property
    def action_dim(self):
        # pick (which D2D pair, which channel)
        return self.M * self.N

    def _cue_power_and_capacity(self, assignment):
        """
        Compute CUE powers meeting SINR threshold against D2D interference:
          Pt_req[i] = (I_d2d[i] + N0) * SINRTh * (r_CUE[i]**alpha) / gain_PL
          Pt[i] = min(Pt_req[i], Pt_cue_max)

        Then the achieved SINR and rate for each CUE under those final powers.
        """
        # Interference at gNB from DTUEs that reuse channel i
        I_d2d = np.zeros(self.N, dtype=float)  # per-channel
        for m in range(self.M):
            ch = assignment[m]
            # received power from DTUE_m at gNB
            I_d2d[ch] += (self.Pt_d2d * self.gain_PL) / ((self.r_DTUE[m] + 1e-9) ** self.alpha)

        Pt_req = (I_d2d + self.N0) * self.SINRTh * (self.r_CUE ** self.alpha) / self.gain_PL
        Pt = np.minimum(Pt_req, self.Pt_cue_max)

        # resulting CUE SINR (with capped power)
        sinr = (Pt * self.gain_PL) / ((self.r_CUE ** self.alpha) * (I_d2d + self.N0) + 1e-12)
        rate = self.B * np.log2(1.0 + sinr)   # Mbps, since B in MHz (consistent relative scale)
        return Pt, rate

    def _due_capacity(self, assignment, CUE_Pt):
        """
        For each D2D pair j on channel m, compute a single-user link capacity:
           signal = Pt_d2d * gain_PL / |DTUE_j - DRUE_j|^alpha
           interference = (CUE_Pt[m] * gain_PL) / |CUE_m - DRUE_j|^alpha + N0
           rate_j = B * log2(1 + signal / interference)
        """
        # Precompute CUE interference to each DRUE if sharing channel m
        inter_cue_to_due = np.zeros((self.M, self.N), dtype=float)
        for j in range(self.M):
            for m in range(self.N):
                d = np.linalg.norm(self.CUE[m] - self.DRUE[j]) + 1e-9
                inter_cue_to_due[j, m] = (CUE_Pt[m] * self.gain_PL) / (d ** self.alpha)

        DUE_rate = np.zeros(self.M, dtype=float)
        for j in range(self.M):
            m = assignment[j]
            d_pair = max(self.r_Dpair[j], 1e-9)
            signal = (self.Pt_d2d * self.gain_PL) / (d_pair ** self.alpha)
            I = inter_cue_to_due[j, m] + self.N0
            DUE_rate[j] = self.B * np.log2(1.0 + signal / I)
        return DUE_rate

    def step(self, action):
        # decode action
        d2d = action // self.N
        ch  = action %  self.N
        self.assign[d2d] = ch

        CUE_Pt, CUE_rate = self._cue_power_and_capacity(self.assign)
        DUE_rate = self._due_capacity(self.assign, CUE_Pt)
        total_capacity = float(np.sum(CUE_rate) + np.sum(DUE_rate))

        # EMA smooth reward to reduce variance for DQN
        self.reward_smooth = self.reward_ema * self.reward_smooth + (1.0 - self.reward_ema) * total_capacity
        # reward scale: heuristic divider to keep roughly O(1)
        reward = self.reward_smooth / 40.0

        return self._feature(), reward, False, {"capacity": total_capacity}

    def _feature(self):
        # normalized + weighted + smoothed feature vector
        ch = self.assign.astype(np.float32) / max(1, (self.N - 1))
        d_g = self.r_DTUE / self.R               # DTUE distance to gNB, normalized
        d_p = self.r_Dpair / self.L              # pair distance normalized by L
        ratio = d_g / (d_p + 1e-6)               # proxy of interference ratio
        inter = ratio / (np.max(ratio) + 1e-12)
        sinr_proxy = np.clip(np.log10(1 + 1.0 / (ratio + 1e-6)), 0.0, 1.5) / 1.5

        F = np.stack([ch, d_g, d_p, inter, sinr_proxy])
        w = np.array([0.2, 0.2, 0.15, 0.25, 0.2])[:, None]
        v = (F * w).reshape(-1).astype(np.float32)

        if self.prev_feature is None:
            s = v
        else:
            s = self.state_ema * self.prev_feature + (1.0 - self.state_ema) * v
        self.prev_feature = s.copy()
        return s

# =====================================================
# Two-Cell Environment
# =====================================================
class TwoCellEnv:
    """
    Two cells (A,B) with their own gNB at (0,0) for simplicity (can be offset),
    each with N_a, N_b CUEs respectively (we keep N_a=N_b=N) and M_a, M_b D2D pairs.
    Frequency reuse across cells: same N subchannels are reused in both cells.
    Inter-cell interference is accounted for when the same subchannel index is used.

    Action: choose a (cell, D2D index in that cell, channel) triple encoded into integer.
    """
    def __init__(self,
                 M_a=8, M_b=8, N=3,
                 R=600.0, L=20.0, cell_offset=1200.0,
                 Pt_d2d=0.01, Pt_cue_max=2.0,
                 B_MHz=0.15, alpha=4.0, N0_dbm=-90.0, SINRTh_db=4.6,
                 gain_pl=2.5e5,
                 state_ema=0.8, reward_ema=0.9,
                 move_sigma_ratio=0.02,
                 seed=3030):
        self.M_a, self.M_b, self.N = int(M_a), int(M_b), int(N)
        self.R, self.L = float(R), float(L)
        self.Pt_d2d = float(Pt_d2d)
        self.Pt_cue_max = float(Pt_cue_max)
        self.B = float(B_MHz)
        self.alpha = float(alpha)
        self.N0 = dbm_to_w(N0_dbm)
        self.SINRTh = db_to_lin(SINRTh_db)
        self.gain_PL = float(gain_pl)
        self.state_ema = float(state_ema)
        self.reward_ema = float(reward_ema)
        self.move_sigma = self.R * float(move_sigma_ratio)
        self.rng = np.random.default_rng(seed)

        # cell B is shifted along +x axis by cell_offset (center-to-center)
        self.offA = np.array([0.0, 0.0])
        self.offB = np.array([float(cell_offset), 0.0])

        self._layout_ready = False
        self.prev_feature = None
        self.reward_smooth = 0.0
        self.episode_counter = 0

    # helpers
    def _uniform_disc(self, n, radius):
        r = np.sqrt(self.rng.random(n)) * radius
        th = self.rng.random(n) * 2.0 * np.pi
        return np.stack([r*np.cos(th), r*np.sin(th)], axis=1)

    def _place_nodes(self):
        # CUEs for both cells
        self.CUE_A = self.offA + self._uniform_disc(self.N, self.R)
        self.CUE_B = self.offB + self._uniform_disc(self.N, self.R)
        # D2D pairs for both cells
        self.DTUE_A = self.offA + self._uniform_disc(self.M_a, self.R)
        self.DTUE_B = self.offB + self._uniform_disc(self.M_b, self.R)
        self.DRUE_A = self.DTUE_A + self._uniform_disc(self.M_a, self.L)
        self.DRUE_B = self.DTUE_B + self._uniform_disc(self.M_b, self.L)

        # distances to their respective gNBs
        self.r_CUE_A = np.linalg.norm(self.CUE_A - self.offA, axis=1)
        self.r_CUE_B = np.linalg.norm(self.CUE_B - self.offB, axis=1)
        self.r_DTUE_A = np.linalg.norm(self.DTUE_A - self.offA, axis=1)
        self.r_DTUE_B = np.linalg.norm(self.DTUE_B - self.offB, axis=1)
        self.r_Dpair_A = np.linalg.norm(self.DTUE_A - self.DRUE_A, axis=1)
        self.r_Dpair_B = np.linalg.norm(self.DTUE_B - self.DRUE_B, axis=1)

        self._layout_ready = True

    def reset(self):
        if not self._layout_ready:
            self._place_nodes()
        else:
            # move either cell A or B entity a bit
            if self.rng.random() < 0.5:
                # move a CUE in A or B
                if self.rng.random() < 0.5:
                    i = self.rng.integers(0, self.N)
                    self.CUE_A[i] += self.rng.normal(0, self.move_sigma, 2)
                    self.r_CUE_A[i] = np.linalg.norm(self.CUE_A[i] - self.offA)
                else:
                    i = self.rng.integers(0, self.N)
                    self.CUE_B[i] += self.rng.normal(0, self.move_sigma, 2)
                    self.r_CUE_B[i] = np.linalg.norm(self.CUE_B[i] - self.offB)
            else:
                # move a D2D in A or B
                if self.rng.random() < 0.5:
                    j = self.rng.integers(0, self.M_a)
                    self.DTUE_A[j] += self.rng.normal(0, self.move_sigma, 2)
                    self.DRUE_A[j] = self.DTUE_A[j] + self._uniform_disc(1, self.L*0.25)[0]
                    self.r_DTUE_A[j] = np.linalg.norm(self.DTUE_A[j] - self.offA)
                    self.r_Dpair_A[j] = np.linalg.norm(self.DTUE_A[j] - self.DRUE_A[j])
                else:
                    j = self.rng.integers(0, self.M_b)
                    self.DTUE_B[j] += self.rng.normal(0, self.move_sigma, 2)
                    self.DRUE_B[j] = self.DTUE_B[j] + self._uniform_disc(1, self.L*0.25)[0]
                    self.r_DTUE_B[j] = np.linalg.norm(self.DTUE_B[j] - self.offB)
                    self.r_Dpair_B[j] = np.linalg.norm(self.DTUE_B[j] - self.DRUE_B[j])

        # random initial reuse
        self.assign_A = self.rng.integers(0, self.N, size=self.M_a)
        self.assign_B = self.rng.integers(0, self.N, size=self.M_b)

        self.prev_feature = None
        self.reward_smooth = 0.0
        self.episode_counter += 1

        return self._feature()

    @property
    def action_dim(self):
        # choose a (cell, d2d_index_in_cell, channel)
        return (self.M_a + self.M_b) * self.N

    # --- CUE power and capacity for a given cell with intra- and inter-cell D2D interference ---
    def _cue_power_and_capacity_cell(self, assign_cell, r_CUE_cell, r_DTUE_cell,
                                     assign_other, r_DTUE_other, other_is_B=False):
        # Interference at this cell's gNB from both local and other-cell DTUEs (same channel index)
        I_d2d = np.zeros(self.N, dtype=float)

        # intra-cell DTUE -> this gNB
        for m in range(len(assign_cell)):
            ch = assign_cell[m]
            I_d2d[ch] += (self.Pt_d2d * self.gain_PL) / ((r_DTUE_cell[m] + 1e-9) ** self.alpha)

        # inter-cell DTUE -> this gNB
        for m in range(len(assign_other)):
            ch = assign_other[m]
            # distance from other cell's DTUE to *this* gNB’s location
            # if other_is_B=False => this is cell A gNB at offA; DTUE_other w.r.t offA
            # we passed r_DTUE_other as distance to other gNB; need absolute positions for cross distance
            # For simplicity, recompute accurately from positions stored in self:
            if other_is_B:
                # we are in cell A; other is B → distance from DTUE_B to offA
                pos = self.DTUE_B[m]
                d_cross = np.linalg.norm(pos - self.offA) + 1e-9
            else:
                # we are in cell B; other is A → distance from DTUE_A to offB
                pos = self.DTUE_A[m]
                d_cross = np.linalg.norm(pos - self.offB) + 1e-9
            I_d2d[ch] += (self.Pt_d2d * self.gain_PL) / (d_cross ** self.alpha)

        Pt_req = (I_d2d + self.N0) * self.SINRTh * (r_CUE_cell ** self.alpha) / self.gain_PL
        Pt = np.minimum(Pt_req, self.Pt_cue_max)
        sinr = (Pt * self.gain_PL) / ((r_CUE_cell ** self.alpha) * (I_d2d + self.N0) + 1e-12)
        rate = self.B * np.log2(1.0 + sinr)
        return Pt, rate

    def _due_capacity_cell(self, assign_cell, DRUE_cell, r_Dpair_cell,
                           CUE_pos_cell, CUE_Pt_cell,
                           assign_other, CUE_pos_other, CUE_Pt_other):
        # interference from CUE (same channel) of own cell + other cell
        DUE_rate = np.zeros(len(assign_cell), dtype=float)
        for j in range(len(assign_cell)):
            m = assign_cell[j]
            d_pair = max(r_Dpair_cell[j], 1e-9)
            sig = (self.Pt_d2d * self.gain_PL) / (d_pair ** self.alpha)
            # own cell CUE_m → DRUE_j
            d1 = np.linalg.norm(CUE_pos_cell[m] - DRUE_cell[j]) + 1e-9
            I = (CUE_Pt_cell[m] * self.gain_PL) / (d1 ** self.alpha)

            # other cell CUE_m (same channel index) → DRUE_j
            d2 = np.linalg.norm(CUE_pos_other[m] - DRUE_cell[j]) + 1e-9
            I += (CUE_Pt_other[m] * self.gain_PL) / (d2 ** self.alpha)

            I += self.N0
            DUE_rate[j] = self.B * np.log2(1.0 + sig / I)
        return DUE_rate

    def step(self, action):
        # decode action over concatenated (A then B)
        cell_block = self.N
        total_pairs = self.M_a + self.M_b
        d2d_idx = action // self.N
        ch = action % self.N
        if d2d_idx < self.M_a:
            self.assign_A[d2d_idx] = ch
        else:
            self.assign_B[d2d_idx - self.M_a] = ch

        # First compute CUE powers for both cells (each sees intra+inter cell D2D)
        Pt_A, rate_CUE_A = self._cue_power_and_capacity_cell(
            self.assign_A, self.r_CUE_A, self.r_DTUE_A,
            self.assign_B, self.r_DTUE_B, other_is_B=True
        )
        Pt_B, rate_CUE_B = self._cue_power_and_capacity_cell(
            self.assign_B, self.r_CUE_B, self.r_DTUE_B,
            self.assign_A, self.r_DTUE_A, other_is_B=False
        )

        # DUE capacities in A and B against both cells' CUE on same channel
        rate_DUE_A = self._due_capacity_cell(
            self.assign_A, self.DRUE_A, self.r_Dpair_A,
            self.CUE_A, Pt_A,
            self.assign_B, self.CUE_B, Pt_B
        )
        rate_DUE_B = self._due_capacity_cell(
            self.assign_B, self.DRUE_B, self.r_Dpair_B,
            self.CUE_B, Pt_B,
            self.assign_A, self.CUE_A, Pt_A
        )

        total_capacity = float(np.sum(rate_CUE_A) + np.sum(rate_CUE_B) +
                               np.sum(rate_DUE_A) + np.sum(rate_DUE_B))

        self.reward_smooth = self.reward_ema * self.reward_smooth + (1.0 - self.reward_ema) * total_capacity
        reward = self.reward_smooth / 80.0  # two-cell scale, keep roughly O(1)

        return self._feature(), reward, False, {"capacity": total_capacity}

    def _feature(self):
        # simple concatenation of both cells' normalized features
        chA = self.assign_A.astype(np.float32) / max(1, (self.N - 1))
        chB = self.assign_B.astype(np.float32) / max(1, (self.N - 1))
        dA = self.r_DTUE_A / self.R
        dB = self.r_DTUE_B / self.R
        pA = self.r_Dpair_A / self.L
        pB = self.r_Dpair_B / self.L
        v = np.concatenate([chA, chB, dA, dB, pA, pB]).astype(np.float32)

        if self.prev_feature is None:
            s = v
        else:
            s = self.state_ema * self.prev_feature + (1.0 - self.state_ema) * v
        self.prev_feature = s.copy()
        return s

# =====================================================
# Agents
# =====================================================
class QLearningAgent:
    def __init__(self, n_actions, alpha=0.3, gamma=0.6, eps=1.0, eps_decay=0.999, eps_min=0.05):
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.eps, self.eps_decay, self.eps_min = float(eps), float(eps_decay), float(eps_min)
        self.n_actions = int(n_actions)
        self.Q = defaultdict(lambda: np.zeros(self.n_actions, dtype=float))

    def _key(self, obs: np.ndarray):
        # discretize observation lightly for tabular key
        # bucket to 2 decimals for stability
        return tuple(np.round(obs.astype(np.float32), 2))

    def act(self, obs: np.ndarray):
        if np.random.random() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[self._key(obs)]))

    def update(self, s, a, r, ns):
        k = self._key(s)
        nk = self._key(ns)
        q = self.Q[k]
        nq = self.Q[nk]
        q[a] += self.alpha * (r + self.gamma * np.max(nq) - q[a])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

if TORCH_AVAILABLE:
    class QNet(nn.Module):
        def __init__(self, sd, ad):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(sd, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, ad)
            )
        def forward(self, x):
            return self.net(x)

# =====================================================
# Training Loops
# =====================================================
def train_compare(env, episodes=400, steps=400, batch=512, gamma=0.6, lr=5e-4, tgt_sync=400):
    """
    Train Q-learning and DQN on the SAME environment distribution (fresh resets),
    and return per-episode average capacity curves and per-step best-so-far capacity.
    """
    # independent environments for QL and DQN to avoid cross-impact of actions
    env_q = env
    # clone-like new env with different seed for DQN
    if isinstance(env, SingleCellEnv):
        env_d = SingleCellEnv(M=env.M, N=env.N, R=env.R, L=env.L,
                              Pt_d2d=env.Pt_d2d, Pt_cue_max=env.Pt_cue_max,
                              B_MHz=env.B, alpha=env.alpha,
                              N0_dbm=10*math.log10(env.N0*1000.0)+30.0,  # reverse to dBm approx
                              SINRTh_db=10*math.log10(env.SINRTh),
                              gain_pl=env.gain_PL, seed=env.rng.integers(1, 1<<31))
    else:
        env_d = TwoCellEnv(M_a=env.M_a, M_b=env.M_b, N=env.N, R=env.R, L=env.L,
                           Pt_d2d=env.Pt_d2d, Pt_cue_max=env.Pt_cue_max,
                           B_MHz=env.B, alpha=env.alpha,
                           N0_dbm=10*math.log10(env.N0*1000.0)+30.0,
                           SINRTh_db=10*math.log10(env.SINRTh),
                           gain_pl=env.gain_PL, seed=env.rng.integers(1, 1<<31))

    # Q-Learning
    s_q = env_q.reset()
    ql = QLearningAgent(env_q.action_dim)
    cap_q = []
    best_q = []
    bestC_q = -1e9

    # DQN
    cap_d = []
    best_d = []
    bestC_d = -1e9
    if TORCH_AVAILABLE:
        s_d = env_d.reset()
        sd = len(s_d)
        ad = env_d.action_dim
        qnet = QNet(sd, ad)
        tgt = QNet(sd, ad)
        tgt.load_state_dict(qnet.state_dict())
        opt = optim.Adam(qnet.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()
        rb = deque(maxlen=100000)
        eps = 1.0; eps_end = 0.05; eps_decay = 0.9995
        step = 0

    for ep in range(episodes):
        s_q = env_q.reset()
        if TORCH_AVAILABLE:
            s_d = env_d.reset()

        ep_cap_q = 0.0
        ep_cap_d = 0.0

        for t in range(steps):
            # --------- Q-Learning ---------
            a_q = ql.act(s_q)
            ns_q, r_q, d_q, info_q = env_q.step(a_q)
            ql.update(s_q, a_q, r_q, ns_q)
            s_q = ns_q
            ep_cap_q += info_q["capacity"]
            bestC_q = max(bestC_q, info_q["capacity"])
            best_q.append(bestC_q)

            # --------- DQN (if available) ---------
            if TORCH_AVAILABLE:
                if np.random.random() < eps:
                    a = np.random.randint(env_d.action_dim)
                else:
                    with torch.no_grad():
                        qv = qnet(torch.tensor(s_d[None, :], dtype=torch.float32))
                        a = int(qv.argmax().item())
                ns_d, r_d, d_d, info_d = env_d.step(a)
                ep_cap_d += info_d["capacity"]
                bestC_d = max(bestC_d, info_d["capacity"])
                best_d.append(bestC_d)

                # store & learn
                rb.append((s_d, a, r_d, ns_d, 0.0))
                s_d = ns_d

                if len(rb) > batch:
                    idx = np.random.choice(len(rb), batch, replace=False)
                    S, A, R, NS, D = zip(*[rb[i] for i in idx])
                    S  = torch.from_numpy(np.array(S, dtype=np.float32))
                    A  = torch.from_numpy(np.array(A, dtype=np.int64)).unsqueeze(1)
                    R  = torch.from_numpy(np.array(R, dtype=np.float32)).unsqueeze(1)
                    NS = torch.from_numpy(np.array(NS, dtype=np.float32))
                    with torch.no_grad():
                        q_next = tgt(NS).max(1, keepdim=True).values
                        Y = R + gamma * q_next
                    q_val = qnet(S).gather(1, A)
                    loss = loss_fn(q_val, Y)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
                    opt.step()
                    step += 1
                    if step % tgt_sync == 0:
                        tgt.load_state_dict(qnet.state_dict())
                eps = max(eps_end, eps * eps_decay)

        cap_q.append(ep_cap_q / steps)
        if TORCH_AVAILABLE:
            cap_d.append(ep_cap_d / steps)

        if (ep + 1) % 50 == 0:
            if TORCH_AVAILABLE:
                print(f"[Ep {ep+1:4d}] QL avg {cap_q[-1]:.2f} | DQN avg {cap_d[-1]:.2f}")
            else:
                print(f"[Ep {ep+1:4d}] QL avg {cap_q[-1]:.2f}")

    return cap_q, best_q, (cap_d if TORCH_AVAILABLE else None), (best_d if TORCH_AVAILABLE else None)

def plot_curves(avg, best, fname_avg, fname_best, label_avg="Average Capacity", label_best="Best-so-far"):
    # Single-plot per figure; do not set colors/styles explicitly
    plt.figure(figsize=(7,4))
    plt.plot(avg, label=label_avg)
    plt.xlabel("Episodes"); plt.ylabel("Capacity (relative Mbps)")
    plt.title(label_avg)
    plt.legend(); plt.tight_layout(); plt.savefig(fname_avg, dpi=140)

    plt.figure(figsize=(7,4))
    plt.plot(best, label=label_best)
    plt.xlabel("Steps"); plt.ylabel("Capacity (relative Mbps)")
    plt.title(label_best)
    plt.legend(); plt.tight_layout(); plt.savefig(fname_best, dpi=140)


def plot_combined_best(s_best, s_dbest, t_best, t_dbest, fname="combined_best.png"):
    # One chart, four curves for step-wise best-so-far capacity; no explicit colors/styles.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(s_best, label="Single-Cell Q-Learning (Best-so-far)")
    if s_dbest is not None:
        plt.plot(s_dbest, label="Single-Cell DQN (Best-so-far)")
    plt.plot(t_best, label="Two-Cell Q-Learning (Best-so-far)")
    if t_dbest is not None:
        plt.plot(t_dbest, label="Two-Cell DQN (Best-so-far)")
    plt.xlabel("Steps")
    plt.ylabel("Best-so-far Capacity (relative Mbps)")
    plt.title("Q-Learning vs DQN — Single vs Two Cell (Best-so-far)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)

def plot_combined_avg(s_avg, s_davg, t_avg, t_davg, fname="combined_avg.png"):
    # One chart, four curves; do not set explicit colors/styles.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(s_avg, label="Single-Cell Q-Learning")
    if s_davg is not None:
        plt.plot(s_davg, label="Single-Cell DQN")
    plt.plot(t_avg, label="Two-Cell Q-Learning")
    if t_davg is not None:
        plt.plot(t_davg, label="Two-Cell DQN")
    plt.xlabel("Episodes")
    plt.ylabel("Average Capacity (Mbps)")
    plt.title("Q-Learning vs DQN — Single vs Two Cell (Episode Average Capacity)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=160)

# =====================================================
# Main
# =====================================================
def main():
    # Default parameters matching the paper's Table 3 where applicable
    single = SingleCellEnv(M=8, N=3, R=600, L=20, Pt_d2d=0.01, Pt_cue_max=2.0,
                           B_MHz=0.15, alpha=4.0, N0_dbm=-90.0, SINRTh_db=4.6,
                           gain_pl=2.5e5, seed=2025)

    two = TwoCellEnv(M_a=8, M_b=8, N=3, R=600, L=20, Pt_d2d=0.01, Pt_cue_max=2.0,
                     B_MHz=0.15, alpha=4.0, N0_dbm=-90.0, SINRTh_db=4.6,
                     gain_pl=2.5e5, seed=3030)

    EPISODES = 1000
    STEPS = 160

    print("=== Training on SINGLE-CELL environment ===")
    s_avg, s_best, s_davg, s_dbest = train_compare(single, episodes=EPISODES, steps=STEPS)
    plot_curves(s_avg, s_best, "single_avg.png", "single_best.png",
                label_avg="Single-Cell: Episode Average Capacity",
                label_best="Single-Cell: Best-so-far Capacity")

    print("=== Training on TWO-CELL environment ===")
    t_avg, t_best, t_davg, t_dbest = train_compare(two, episodes=EPISODES, steps=STEPS)
    plot_curves(t_avg, t_best, "twocell_avg.png", "twocell_best.png",
                label_avg="Two-Cell: Episode Average Capacity",
                label_best="Two-Cell: Best-so-far Capacity")

    # Combined single figures with 4 results
    plot_combined_avg(s_avg, s_davg, t_avg, t_davg, fname="combined_avg.png")
    plot_combined_best(s_best, s_dbest, t_best, t_dbest, fname="combined_best.png")

    print("Saved figures: single_avg.png, single_best.png, twocell_avg.png, twocell_best.png, combined_avg.png, combined_best.png")

if __name__ == "__main__":
    main()
