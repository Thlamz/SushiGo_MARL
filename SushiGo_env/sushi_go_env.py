"""
Sushi Go! — PettingZoo Parallel environment for multi-agent reinforcement learning.

Sushi Go is a card-drafting game. Over 3 rounds, all players SIMULTANEOUSLY pick one
card from their hand, then pass the rest of the hand to the next player. After each
round, dishes are scored; pudding is scored once at the end of the game.

Because every player moves at the same time, this is modelled with the PettingZoo
Parallel API (the natural fit for simultaneous-move games). Convert to the AEC
paradigm with `pettingzoo.utils.parallel_to_aec` if your training stack needs it.


OBSERVATION DESIGN

Each agent's observation is a flat float vector with four sections, plus an
`action_mask`. The observation is built ego-centrically (from that agent's seat).

  1. current_hand        : counts of each of the 12 card types in the hand the
                           agent is holding RIGHT NOW (the hand it drafts from).

  2. hand_history        : a rolling memory of the last `history_len` hands the
                           agent has seen this round, most-recent-first. Each
                           remembered hand is a 12-vector of card-type counts,
                           recorded AS THE AGENT RECEIVED IT (before it drafted).
                           This is imperfect information: slot k holds the hand
                           seen k turns ago, and since then k other players have
                           drafted from it, so that memory is k drafts STALE.
                           `history_len` defaults to `n_players - 1`, i.e. exactly
                           one lap of the table — after that the hand would have
                           come all the way back around and the memory is
                           completely outdated, so it drops out of the buffer.
                           Slots are zero-filled at the start of each round before
                           the buffer has filled.

  3. own_tableau         : the agent's own cards on the table — 12 type counts
                           plus [unused_wasabi, pudding_total].

  4. opponent_tableaus   : every opponent's cards on the table (public info in
                           Sushi Go), ordered by seat offset, each as 12 counts
                           plus [unused_wasabi, pudding_total]. Disable with
                           `include_opponent_tableaus=False`.

   5. cards_played    : counts of each of the 12 card types descarted by all
                            players so far this round 

  + game_scalars         : [round_index / 3, cards_in_hand / hand_size].

The observation size depends on `n_players` and `history_len` (a fresh env is
created per player count for training). `split_observation()` decodes a raw
vector back into named sections for debugging.


ACTION DESIGN
Discrete(12): choose which CARD TYPE to draft this turn. Working in card-type
space (rather than hand-position space) gives a fixed, semantically meaningful
action space and a trivial action mask (`mask[t] = 1` iff a card of type t is in
the current hand). Invalid actions are gracefully redirected to a legal card.


CHOPSTICKS
Chopsticks are kept in the deck as an authentic collectible card, but the
"swap chopsticks back to draft 2 cards in one turn" mechanic is OMITTED in this
version so the action stays a single Discrete. To add it, turn the action into a
MultiDiscrete([12, 13]) where the second component is 0 (no chopsticks) or
1..12 (also draft that type, returning a chopsticks token to the hand).
"""

import functools
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as DictSpace
from pettingzoo import ParallelEnv


TEMPURA, SASHIMI, DUMPLING = 0, 1, 2
MAKI1, MAKI2, MAKI3 = 3, 4, 5
NIGIRI_EGG, NIGIRI_SALMON, NIGIRI_SQUID = 6, 7, 8
WASABI, PUDDING, CHOPSTICKS = 9, 10, 11
N_TYPES = 12

CARD_NAMES = [
    "Tempura", "Sashimi", "Dumpling", "Maki1", "Maki2", "Maki3",
    "Nigiri-Egg", "Nigiri-Salmon", "Nigiri-Squid", "Wasabi", "Pudding", "Chopsticks",
]

# Authentic 108-card Sushi Go! deck composition.
DECK_COMPOSITION = {
    TEMPURA: 14, SASHIMI: 14, DUMPLING: 14,
    MAKI1: 6, MAKI2: 12, MAKI3: 8,
    NIGIRI_EGG: 5, NIGIRI_SALMON: 10, NIGIRI_SQUID: 5,
    WASABI: 6, PUDDING: 10, CHOPSTICKS: 4,
}
assert sum(DECK_COMPOSITION.values()) == 108

MAKI_ICONS = {MAKI1: 1, MAKI2: 2, MAKI3: 3}
NIGIRI_VALUE = {NIGIRI_EGG: 1, NIGIRI_SALMON: 2, NIGIRI_SQUID: 3}
DUMPLING_SCORE = [0, 1, 3, 6, 10, 15]  # index = dumpling count, clamped at 5

MAX_PLAYERS = 4
N_ROUNDS = 3


def hand_size_for(n_players: int) -> int:
    """Cards dealt per player per round. 2p:10, 3p:9, 4p:8 (12 - n_players)."""
    return 12 - n_players


# Environment
class SushiGoParallelEnv(ParallelEnv):
    """Sushi Go! as a PettingZoo ParallelEnv (2-4 players)."""

    metadata = {"render_modes": ["human"], "name": "sushi_go_v2", "is_parallelizable": True}

    def __init__(
        self,
        n_players: int | None = 3,
        min_n_players: int | None = None,
        max_n_players: int | None = None,
        history_len: int | None = None,
        include_opponent_tableaus: bool = True,
        reward_scale: float = 1.0,
        render_mode=None,
    ):
        if n_players is not None and (min_n_players is not None or max_n_players is not None):
            raise ValueError("Use either n_players or min_n_players/max_n_players, not both.")
        if n_players is not None:
            min_n_players = max_n_players = n_players
        if min_n_players is None:
            min_n_players = 3
        if max_n_players is None:
            max_n_players = min_n_players
        assert 2 <= min_n_players <= max_n_players <= MAX_PLAYERS, (
            "player count bounds must satisfy 2 <= min_n_players <= max_n_players <= 4"
        )
        super().__init__()

        self.min_n_players = min_n_players
        self.max_n_players = max_n_players
        self.n_players = max_n_players
        self.active_n_players = max_n_players
        self.hand_size = hand_size_for(self.active_n_players)
        # One lap of the table = n_players hands; the n_players-1 *previous* hands
        # span exactly up to the point where the info becomes fully outdated.
        self.history_len = (max_n_players - 1) if history_len is None else history_len
        assert self.history_len >= 0
        self.include_opponent_tableaus = include_opponent_tableaus
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        self.last_rewards = np.zeros(self.max_n_players, dtype=np.float64)
        self.cards_discarted = np.zeros(N_TYPES, dtype=np.int64)
        self.player_mask = np.ones(self.max_n_players, dtype=bool)

        self.possible_agents = [f"player_{i}" for i in range(max_n_players)]
        self.agents = list(self.possible_agents)

        # observation layout (named slices into the flat vector) 
        n_opp = (max_n_players - 1) if include_opponent_tableaus else 0
        sizes = [
            ("current_hand", N_TYPES),                       
            ("hand_history", self.history_len * N_TYPES),    
            ("own_tableau", N_TYPES + 2),                    
            ("opponent_tableaus", n_opp * (N_TYPES + 2)),
            ("cards_played", N_TYPES),   
            ("game_scalars", 2),
        ]
        self.obs_slices, cursor = {}, 0
        for name, size in sizes:
            self.obs_slices[name] = (cursor, cursor + size)
            cursor += size
        self.obs_dim = cursor

        self.rng = np.random.default_rng()
        self.observation_spaces = {a: self.observation_space(a) for a in self.possible_agents}
        self.action_spaces = {a: self.action_space(a) for a in self.possible_agents}

    #  spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return DictSpace({
            "observation": Box(low=-1.0, high=50.0, shape=(self.obs_dim,), dtype=np.float32),
            "action_mask": Box(low=0, high=1, shape=(N_TYPES,), dtype=np.int8),
            "player_mask": Box(low=0, high=1, shape=(), dtype=bool),
        })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_TYPES)

    # core lifecycle
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.active_n_players = int(self.rng.integers(self.min_n_players, self.max_n_players + 1))
        self.n_players = self.active_n_players
        self.hand_size = hand_size_for(self.active_n_players)
        self.player_mask = np.arange(self.max_n_players) < self.active_n_players

        deck = []
        for card, count in DECK_COMPOSITION.items():
            deck += [card] * count
        self.deck = [int(x) for x in self.rng.permutation(deck)]

        self.pudding_total = np.zeros(self.max_n_players, dtype=np.int64)
        self.round_idx = 1
        self.turn = 0
        self.last_rewards = np.zeros(self.max_n_players, dtype=np.float64)
        self.cards_discarted = np.zeros(N_TYPES, dtype=np.int64)
        self._deal_round()  # deals hands; clears tableaus and hand-history memory

        observations = {a: self._obs_for(i) for i, a in enumerate(self.agents)}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        acting = list(self.agents)

        # Snapshot each hand AS SEEN this turn (before anyone drafts) — this is what
        # players will remember in their hand_history.
        snapshots = [self._hand_counts(p) for p in range(self.active_n_players)]

        # Each player drafts one card of the chosen type.
        for p, agent in enumerate(acting[:self.active_n_players]):
            a = int(actions[agent])
            hand = self.hands[p]
            if a not in hand:  # graceful fallback for invalid (unmasked) actions
                a = hand[0]
            hand.remove(a)
            self._place_card(p, a)
        self.turn += 1

        rewards = {a: 0.0 for a in acting}
        terminations = {a: False for a in acting}
        truncations = {a: False for a in acting}
        infos = {a: {} for a in acting}

        round_over = len(self.hands[0]) == 0
        current_scores = self._score_turn()
        turn_scores = current_scores - self.last_rewards        
        for p, agent in enumerate(acting[:self.active_n_players]):
                rewards[agent] += float(turn_scores[p])
                infos[agent]["turn_score"] = float(turn_scores[p])
        self.last_rewards = current_scores

        if not round_over:
            # Each player files away the hand it just saw, then hands pass one seat
            # along: new_hands[i] = old_hands[i-1].
            for p in range(self.active_n_players):
                self.seen_history[p].insert(0, snapshots[p])
                del self.seen_history[p][self.history_len:]
            self.hands[:self.active_n_players] = [
                self.hands[(i - 1) % self.active_n_players]
                for i in range(self.active_n_players)
            ]
        else:            
            if self.round_idx < N_ROUNDS:
                self.round_idx += 1
                self._deal_round()  # clears tableaus + hand-history, deals new hands
            else:
                pud_scores = self._score_pudding()  # pudding scored once, at game end
                for p, agent in enumerate(acting[:self.active_n_players]):
                    rewards[agent] += float(pud_scores[p])
                    infos[agent]["pudding_score"] = float(pud_scores[p])
                terminations = {a: True for a in acting}
                self.agents = []

        rewards = {a: r * self.reward_scale for a, r in rewards.items()}
        observations = {a: self._obs_for(p) for p, a in enumerate(acting)}
        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    # dealing & card placement 
    def _deal_round(self):
        """Clear per-round state (tableaus + hand-history) and deal fresh hands."""
        self.tableau = [np.zeros(N_TYPES, dtype=np.int64) for _ in range(self.max_n_players)]
        self.nigiri_on_wasabi = [np.zeros(N_TYPES, dtype=np.int64)
                                 for _ in range(self.max_n_players)]
        self.wasabi_unused = [0 for _ in range(self.max_n_players)]
        self.seen_history = [[] for _ in range(self.max_n_players)]  # hand memory resets
        self.last_rewards = np.zeros(self.max_n_players, dtype=np.float64)
        self.hands = [
            [self.deck.pop() for _ in range(self.hand_size)] if p < self.active_n_players else []
            for p in range(self.max_n_players)
        ]

    def _place_card(self, p, card):
        """Add a drafted card to player p's tableau, handling pudding and wasabi."""
        self.cards_discarted[card] += 1
        if card == PUDDING:
            self.pudding_total[p] += 1
        elif card in NIGIRI_VALUE:
            self.tableau[p][card] += 1
            if self.wasabi_unused[p] > 0:  # nigiri lands on an unused wasabi -> 3x
                self.wasabi_unused[p] -= 1
                self.nigiri_on_wasabi[p][card] += 1
        elif card == WASABI:
            self.tableau[p][card] += 1
            self.wasabi_unused[p] += 1
        else:  # tempura, sashimi, dumpling, maki, chopsticks
            self.tableau[p][card] += 1

    def _hand_counts(self, p):
        c = np.zeros(N_TYPES, dtype=np.float32)
        for card in self.hands[p]:
            c[card] += 1
        return c

    # scoring 
    def _score_turn(self):
        """Score tempura / sashimi / dumpling / nigiri+wasabi / maki for the round."""
        n = self.active_n_players
        scores = np.zeros(self.max_n_players, dtype=np.float64)
        for p in range(n):
            t = self.tableau[p]
            scores[p] += (t[TEMPURA] // 2) * 5            # 2 tempura -> 5
            scores[p] += (t[SASHIMI] // 3) * 10           # 3 sashimi -> 10
            scores[p] += DUMPLING_SCORE[min(int(t[DUMPLING]), 5)]
            for nt, base in NIGIRI_VALUE.items():         # nigiri, tripled on wasabi
                on_w = self.nigiri_on_wasabi[p][nt]
                scores[p] += (t[nt] - on_w) * base + on_w * base * 3

        maki_counts = np.array([
            sum(self.tableau[p][m] * icons for m, icons in MAKI_ICONS.items())
            for p in range(n)
        ])
        scores[:n] += self._score_maki(maki_counts)
        return scores

    def _score_round(self):
        """Backward-compatible alias for tests/scripts using the old name."""
        return self._score_turn()

    @staticmethod
    def _score_maki(counts):
        """Most maki icons -> 6 pts; runner-up -> 3 pts. Ties split (floored)."""
        n = len(counts)
        out = np.zeros(n, dtype=np.float64)
        top = counts.max()
        if top == 0:
            return out
        firsts = [i for i in range(n) if counts[i] == top]
        for i in firsts:
            out[i] += 6 // len(firsts)
        if len(firsts) == 1:  # a clear winner -> award second place
            rest = [counts[i] for i in range(n) if i not in firsts]
            second = max(rest) if rest else 0
            if second > 0:
                seconds = [i for i in range(n) if counts[i] == second and i not in firsts]
                for i in seconds:
                    out[i] += 3 // len(seconds)
        return out

    def _score_pudding(self):
        """End-of-game pudding: most -> +6, least -> -6 (no penalty in a 2-player game)."""
        n = self.active_n_players
        out = np.zeros(self.max_n_players, dtype=np.float64)
        pud = self.pudding_total[:n]
        most = [i for i in range(n) if pud[i] == pud.max()]
        for i in most:
            out[i] += 6 // len(most)
        if n > 2:  # least-pudding penalty does not apply with 2 players
            least = [i for i in range(n) if pud[i] == pud.min()]
            for i in least:
                out[i] -= 6 // len(least)
        return out

    # ---- observations -----------------------------------------------------------------
    def _tableau_block(self, p):
        """12 type counts + [unused_wasabi, pudding_total] for player p."""
        return np.concatenate([
            self.tableau[p].astype(np.float32),
            np.array([self.wasabi_unused[p], self.pudding_total[p]], dtype=np.float32),
        ])

    def _obs_for(self, p):
        """Ego-centric observation for player p."""
        if p >= self.active_n_players:
            obs = np.full(self.obs_dim, -1.0, dtype=np.float32)
            mask = np.zeros(N_TYPES, dtype=np.int8)
            mask[0] = 1
            return {"observation": obs, "action_mask": mask, "player_mask": np.bool_(False)}

        parts = [self._hand_counts(p)]  # 1. current hand

        # 2. hand history: last `history_len` hands seen, most-recent-first, padded.
        hist = self.seen_history[p]
        for k in range(self.history_len):
            parts.append(hist[k] if k < len(hist) else np.zeros(N_TYPES, dtype=np.float32))

        # 3. own tableau
        parts.append(self._tableau_block(p))

        # 4. opponents' tableaus, ordered by seat offset (ego-centric)
        if self.include_opponent_tableaus:
            for off in range(1, self.max_n_players):
                if off < self.active_n_players:
                    parts.append(self._tableau_block((p + off) % self.active_n_players))
                else:
                    parts.append(np.zeros(N_TYPES + 2, dtype=np.float32))

        # 5. cards_discarted
        DECK_COUNTS = np.array([DECK_COMPOSITION[i] for i in range(N_TYPES)], dtype=np.float32)
        parts.append(self.cards_discarted / DECK_COUNTS)

        # game scalars
        parts.append(np.array([
            self.round_idx / N_ROUNDS,
            len(self.hands[p]) / self.hand_size,
        ], dtype=np.float32))

        obs = np.concatenate(parts).astype(np.float32)
        assert obs.shape[0] == self.obs_dim, (obs.shape[0], self.obs_dim)
        mask = (self._hand_counts(p) > 0).astype(np.int8)
        return {"observation": obs, "action_mask": mask, "player_mask": np.bool_(True)}

    def split_observation(self, obs_vector):
        """Decode a flat observation vector into its named sections (debugging aid)."""
        return {name: np.asarray(obs_vector)[s:e] for name, (s, e) in self.obs_slices.items()}

    # ---- misc -------------------------------------------------------------------------
    def render(self):
        lines = [f"--- Round {self.round_idx} | turn {self.turn} ---"]
        for p in range(self.active_n_players):
            tab = ", ".join(f"{CARD_NAMES[c]}x{int(self.tableau[p][c])}"
                            for c in range(N_TYPES) if self.tableau[p][c] > 0)
            lines.append(f"  player_{p}: [{tab}]  pudding={self.pudding_total[p]}")
        print("\n".join(lines))

    def close(self):
        pass


def env(**kwargs):
    """Factory returning a fresh Sushi Go parallel environment."""
    return SushiGoParallelEnv(**kwargs)


# Random-game demo 
if __name__ == "__main__":
    for n in (2, 3, 4):
        e = SushiGoParallelEnv(n_players=n)
        obs, _ = e.reset(seed=0)
        totals = {a: 0.0 for a in e.possible_agents}
        steps = 0
        while e.agents:
            acts = {}
            for a in e.agents:
                valid = np.flatnonzero(obs[a]["action_mask"])
                acts[a] = int(e.rng.choice(valid))  # masked-random policy
            obs, rew, term, trunc, _ = e.step(acts)
            for a, r in rew.items():
                totals[a] += r
            steps += 1
        print(f"n_players={n}: obs_dim={e.obs_dim}, history_len={e.history_len}, "
              f"{steps} turns, scores: "
              + ", ".join(f"{a}={totals[a]:.0f}" for a in totals))
