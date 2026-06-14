"""Sanity checks: scoring rules, PettingZoo API compliance, hand-history memory."""
import numpy as np
from pettingzoo.test import parallel_api_test
from SushiGo_env.sushi_go_env import (
    SushiGoParallelEnv, TEMPURA, SASHIMI, DUMPLING, NIGIRI_EGG, NIGIRI_SALMON,
    NIGIRI_SQUID, WASABI, N_TYPES, OBS_COMPONENTS, PADDING_VALUE,
)


def test_dish_scoring():
    e = SushiGoParallelEnv(n_players=2)
    e.reset(seed=1)
    e.tableau[0][:] = 0
    e.tableau[0][TEMPURA] = 2     # 2 tempura -> 5
    e.tableau[0][SASHIMI] = 3     # 3 sashimi -> 10
    e.tableau[0][DUMPLING] = 4    # 4 dumplings -> 10
    e.tableau[1][:] = 0
    e.tableau[1][NIGIRI_SQUID] = 1
    e.tableau[1][WASABI] = 1
    e.nigiri_on_wasabi[1][NIGIRI_SQUID] = 1   # squid on wasabi -> 3*3 = 9
    s = e._score_round()
    assert s[0] == 25 and s[1] == 9, s
    print("dish scoring OK            ->", s)


def test_maki_scoring():
    e = SushiGoParallelEnv(n_players=3)
    out = e._score_maki(np.array([5, 3, 1]))      # clear 1st/2nd
    assert list(out) == [6, 3, 0], out
    out2 = e._score_maki(np.array([4, 4, 2]))     # tie for 1st: split 6, no 2nd
    assert list(out2) == [3, 3, 0], out2
    print("maki scoring OK            ->", out, "/", out2)


def test_pudding_scoring():
    e = SushiGoParallelEnv(n_players=4)
    e.pudding_total = np.array([3, 1, 1, 0])
    out = e._score_pudding()                      # most +6, least -6
    assert out[0] == 6 and out[3] == -6 and out[1] == 0, out
    e2 = SushiGoParallelEnv(n_players=2)
    e2.pudding_total = np.array([2, 0])
    out2 = e2._score_pudding()                    # 2 players: no penalty for least
    assert out2[0] == 6 and out2[1] == 0, out2
    print("pudding scoring OK         ->", out, "/", out2)


def test_wasabi_ordering():
    e = SushiGoParallelEnv(n_players=2)
    e.reset(seed=1)
    e._place_card(0, NIGIRI_EGG)      # before wasabi -> normal (1)
    e._place_card(0, WASABI)
    e._place_card(0, NIGIRI_SALMON)   # after wasabi  -> tripled (6)
    assert e._score_round()[0] == 7
    print("wasabi ordering OK         -> nigiri score = 7")


def test_observation_layout():
    for n in (2, 3, 4):
        e = SushiGoParallelEnv(n_players=n)
        obs, _ = e.reset(seed=0)
        o = obs["player_0"]
        for key in OBS_COMPONENTS:
            assert o[key].shape == e.obs_shapes[key]
        flat = e.flatten_observation(o)
        assert flat.shape[0] == e.obs_dim
        assert o["action_mask"].shape[0] == N_TYPES and o["action_mask"].sum() >= 1
        sec = e.split_observation(flat)
        # current_hand counts must equal hand_size at the start of a round
        assert sec["current_hand"].sum() == e.hand_size
        # hand_history starts padded at the start of a round
        assert np.all(sec["hand_history"] == PADDING_VALUE)
        # the action mask matches the current_hand section
        assert np.array_equal((sec["current_hand"] > 0).astype(np.int8), o["action_mask"])
    print("observation layout OK      -> obs_dim 2p/3p/4p =",
          [SushiGoParallelEnv(n_players=n).obs_dim for n in (2, 3, 4)])


def test_stochastic_player_count_reset():
    e = SushiGoParallelEnv(n_players=None, min_n_players=2, max_n_players=4)
    seen_counts = set()
    for seed in range(20):
        obs, _ = e.reset(seed=seed)
        seen_counts.add(e.active_n_players)
        assert len(e.possible_agents) == 4
        assert len(e.agents) == 4
        assert e.player_mask.tolist() == [i < e.active_n_players for i in range(4)]
        for i, agent in enumerate(e.possible_agents):
            o = obs[agent]
            assert e.flatten_observation(o).shape[0] == e.obs_dim
            assert bool(o["player_mask"]) is (i < e.active_n_players)
            assert o["action_mask"].sum() >= 1
            if i >= e.active_n_players:
                for key in OBS_COMPONENTS:
                    assert np.all(o[key] == PADDING_VALUE)
                assert o["action_mask"][0] == 1 and o["action_mask"].sum() == 1
    assert seen_counts <= {2, 3, 4}
    assert len(seen_counts) > 1
    print("stochastic reset OK        -> sampled counts =", sorted(seen_counts))


def test_hand_history_memory():
    """history slot 0 must equal the hand the player held on the previous turn."""
    e = SushiGoParallelEnv(n_players=3)
    obs, _ = e.reset(seed=42)

    # Hand player_0 is holding right now (turn 1).
    hand_t1 = obs["player_0"]["current_hand"].copy()

    # Everyone drafts a legal card; advance one turn.
    acts = {a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in e.agents}
    obs, *_ = e.step(acts)

    slot0 = obs["player_0"]["hand_history"][0]          # most-recent remembered hand
    assert np.array_equal(slot0, hand_t1), (slot0, hand_t1)

    # Advance again: the turn-1 hand shifts to slot 1 (now 2 drafts stale).
    acts = {a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in e.agents}
    obs, *_ = e.step(acts)
    slot1 = obs["player_0"]["hand_history"][1]
    assert np.array_equal(slot1, hand_t1), (slot1, hand_t1)
    print("hand-history memory OK     -> remembered hand shifts down the buffer")


def test_history_resets_each_round():
    e = SushiGoParallelEnv(n_players=3)
    obs, _ = e.reset(seed=7)
    # Play out a full round (hand_size turns).
    for _ in range(e.hand_size):
        acts = {a: int(np.flatnonzero(obs[a]["action_mask"])[0]) for a in e.agents}
        obs, *_ = e.step(acts)
    # New round dealt: hand-history memory must be cleared.
    assert np.all(obs["player_0"]["hand_history"] == PADDING_VALUE)
    assert e.round_idx == 2
    print("history resets each round  -> memory cleared on new deal")


def test_api_compliance():
    for n in (2, 3, 4):
        parallel_api_test(SushiGoParallelEnv(n_players=n), num_cycles=200)
    print("PettingZoo parallel_api_test OK (n_players = 2, 3, 4)")


if __name__ == "__main__":
    test_dish_scoring()
    test_maki_scoring()
    test_pudding_scoring()
    test_wasabi_ordering()
    test_observation_layout()
    test_stochastic_player_count_reset()
    test_hand_history_memory()
    test_history_resets_each_round()
    test_api_compliance()
    print("\nAll checks passed.")
