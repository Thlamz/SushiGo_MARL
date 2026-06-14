"""
Play Sushi Go! against your trained DQN  — local web interface.

You take one seat at the table; your trained Q-network controls the rest. Run this,
open the printed URL in a browser, and play: your hand is shown as clickable cards,
every player's tableau is laid out on the board, scores update each turn.

USAGE
=====
    python3 test/play.py --model models/DQN/sushi_go_qnet_3_players.pt --n-players 3

  --n-players MUST match the player count the model was trained on (the observation
  size depends on it). Default 3, which matches train_madqn.py's default.
  If the model file is missing or fails to load, the app falls back to RANDOM
  opponents and says so in the UI, so you can still play / test the interface.

This reuses build_qvalue_actor (train_dqn.py) and the grouped tensordict keys
(torchrl_integration.py), so model loading is guaranteed consistent with training.

Change this to import to your own trained model
"""
import argparse
import sys

import numpy as np
from flask import Flask, jsonify, request, Response
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ADD THIS LINE: Expose the inside of SushiGo_env to PyTorch's unpickler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../SushiGo_env')))

from SushiGo_env.sushi_go_env import SushiGoParallelEnv, N_TYPES

HUMAN_SEAT = 0

# Card metadata for the UI: emoji, short label, accent colour, maki-roll badge.
CARD_META = [
    {"emoji": "\U0001F364", "label": "Tempura",  "color": "#E0A458", "rolls": 0},
    {"emoji": "\U0001F363", "label": "Sashimi",  "color": "#E8553F", "rolls": 0},
    {"emoji": "\U0001F95F", "label": "Dumpling", "color": "#C9A227", "rolls": 0},
    {"emoji": "\U0001F359", "label": "Maki",     "color": "#3F7D5B", "rolls": 1},
    {"emoji": "\U0001F359", "label": "Maki",     "color": "#3F7D5B", "rolls": 2},
    {"emoji": "\U0001F359", "label": "Maki",     "color": "#3F7D5B", "rolls": 3},
    {"emoji": "\U0001F95A", "label": "Egg",      "color": "#D98E04", "rolls": 0},
    {"emoji": "\U0001F41F", "label": "Salmon",   "color": "#E8553F", "rolls": 0},
    {"emoji": "\U0001F991", "label": "Squid",    "color": "#8E5BA6", "rolls": 0},
    {"emoji": "\U0001F336\uFE0F", "label": "Wasabi", "color": "#4C9A52", "rolls": 0},
    {"emoji": "\U0001F36E", "label": "Pudding",  "color": "#7A5C3E", "rolls": 0},
    {"emoji": "\U0001F962", "label": "Chopstix", "color": "#5C6B7A", "rolls": 0},
]


# ======================================================================================
# Model loading + inference
# ======================================================================================
def load_model(n_players, obs_dim, path):
    """Load the trained DQN Q-network. Returns None on failure (caller falls back)."""
    try:
        import torch
        from tensordict import TensorDict
        from train.train_dqn import build_qvalue_actor
        from torchrl_integration import OBS_KEY, MASK_KEY

        actor = build_qvalue_actor(n_players, obs_dim, device="cpu")
        # Warm up lazy parameters with a dummy observation before loading weights.
        dummy = TensorDict({
            OBS_KEY: torch.zeros(n_players, obs_dim),
            MASK_KEY: torch.ones(n_players, N_TYPES, dtype=torch.bool),
        }, batch_size=[])
        actor(dummy)
        actor.load_state_dict(torch.load(path, map_location="cpu"))
        actor.eval()
        print(f"[model] loaded '{path}'  (n_players={n_players}, obs_dim={obs_dim})")
        return actor
    except FileNotFoundError:
        print(f"[model] '{path}' not found — opponents will play RANDOMLY.")
    except Exception as e:  # noqa: BLE001 — any load error -> graceful fallback
        print(f"[model] failed to load '{path}' ({e}) — opponents will play RANDOMLY.")
    return None


def model_actions(actor, obs_dict, agents):
    """Greedy (masked-argmax) actions from the Q-network for every seat."""
    import torch
    from tensordict import TensorDict
    from torchrl_integration import OBS_KEY, MASK_KEY, ACTION_KEY

    obs = np.stack([obs_dict[a]["observation"] for a in agents]).astype(np.float32)
    mask = np.stack([obs_dict[a]["action_mask"] for a in agents]).astype(bool)
    td = TensorDict({
        OBS_KEY: torch.from_numpy(obs),
        MASK_KEY: torch.from_numpy(mask),
    }, batch_size=[])
    with torch.no_grad():
        actor(td)
    return td[ACTION_KEY].cpu().numpy().astype(int)


def random_actions(rng, obs_dict, agents):
    """Masked-random fallback when no model is available."""
    return [int(rng.choice(np.flatnonzero(obs_dict[a]["action_mask"]))) for a in agents]


# ======================================================================================
# Game session — wraps one env, human is seat 0, model controls the rest
# ======================================================================================
class GameSession:
    def __init__(self, n_players, model_actor):
        self.n_players = n_players
        self.model = model_actor
        self.env = SushiGoParallelEnv(n_players=n_players, reward_scale=1.0)
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset()
        self.totals = {a: 0.0 for a in self.env.possible_agents}
        self.last_draft = {}     # agent -> card type drafted last turn
        self.done = False

    def human_move(self, card_type):
        """Apply the human's draft + the model's drafts for every other seat."""
        if self.done:
            return
        agents = list(self.env.agents)

        if self.model is not None:
            chosen = model_actions(self.model, self.obs, agents)
        else:
            chosen = random_actions(self.rng, self.obs, agents)

        actions = {}
        for i, a in enumerate(agents):
            if i == HUMAN_SEAT:
                actions[a] = int(card_type)        # the human's choice overrides
            else:
                actions[a] = int(chosen[i])
        self.last_draft = dict(actions)

        self.obs, rewards, terminations, _, _ = self.env.step(actions)
        for a, r in rewards.items():
            self.totals[a] += r                    # per-turn rewards sum to final score
        if not self.env.agents:
            self.done = True

    def _winners(self):
        """Highest score wins; ties broken by most pudding (real Sushi Go rule)."""
        best = max(self.totals.values())
        cands = [i for i, a in enumerate(self.env.possible_agents)
                 if self.totals[a] == best]
        if len(cands) > 1:
            best_pud = max(int(self.env.pudding_total[i]) for i in cands)
            cands = [i for i in cands if int(self.env.pudding_total[i]) == best_pud]
        return cands

    def state(self):
        env = self.env
        players = []
        for i, a in enumerate(env.possible_agents):
            players.append({
                "name": "You" if i == HUMAN_SEAT else f"Bot {i}",
                "is_human": i == HUMAN_SEAT,
                "tableau": [int(x) for x in env.tableau[i]],
                "wasabi_open": int(env.wasabi_unused[i]),
                "pudding": int(env.pudding_total[i]),
                "score": round(float(self.totals[a]), 1),
                "last_draft": (self.last_draft.get(a) if self.last_draft else None),
            })
        human_hand = list(env.hands[HUMAN_SEAT]) if not self.done else []
        return {
            "round": int(env.round_idx),
            "turn": int(env.turn),
            "n_rounds": 3,
            "done": self.done,
            "model_ok": self.model is not None,
            "players": players,
            "hand": [int(c) for c in human_hand],
            "winners": self._winners() if self.done else [],
            "card_meta": CARD_META,
        }


# ======================================================================================
# Flask app
# ======================================================================================
app = Flask(__name__)
SESSION: GameSession | None = None   # single local session — one player at a time


@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    SESSION.reset()
    return jsonify(SESSION.state())


@app.route("/api/move", methods=["POST"])
def move():
    card = int(request.get_json(force=True)["card"])
    SESSION.human_move(card)
    return jsonify(SESSION.state())


@app.route("/api/state")
def state():
    return jsonify(SESSION.state())


# ======================================================================================
# Front-end (single inline page)
# ======================================================================================
PAGE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Sushi Go! vs DQN</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Mochiy+Pop+One&family=Zen+Maru+Gothic:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#FBF3E4; --paper-deep:#F3E6CC; --ink:#2A3D66; --ink-soft:#5A6B8C;
    --coral:#E8553F; --sage:#3F7D5B; --gold:#E0A458; --line:#E3D4B4;
    --card:#FFFFFF; --shadow:rgba(42,61,102,.18);
  }
  *{box-sizing:border-box;margin:0;padding:0;}
  body{
    font-family:'Zen Maru Gothic',sans-serif; color:var(--ink);
    background:
      radial-gradient(circle at 12% 8%, #FFFBF0 0%, transparent 45%),
      radial-gradient(circle at 88% 92%, #F6E9CE 0%, transparent 50%),
      var(--paper);
    background-attachment:fixed; min-height:100vh; padding:22px 16px 60px;
  }
  /* faint dotted texture */
  body::before{
    content:""; position:fixed; inset:0; pointer-events:none; opacity:.4;
    background-image:radial-gradient(var(--line) 1px, transparent 1px);
    background-size:22px 22px;
  }
  .wrap{max-width:980px; margin:0 auto; position:relative;}

  header{
    display:flex; align-items:center; justify-content:space-between;
    gap:14px; flex-wrap:wrap; margin-bottom:20px;
  }
  h1{
    font-family:'Mochiy Pop One',sans-serif; font-size:30px; line-height:1;
    color:var(--ink); letter-spacing:.5px;
  }
  h1 .go{color:var(--coral);}
  .badges{display:flex; gap:10px; align-items:center; flex-wrap:wrap;}
  .badge{
    background:var(--card); border:2px solid var(--line); border-radius:14px;
    padding:8px 14px; font-weight:700; font-size:14px;
    box-shadow:0 3px 0 var(--line);
  }
  .badge b{color:var(--coral); font-family:'Mochiy Pop One',sans-serif;}
  .btn{
    font-family:'Mochiy Pop One',sans-serif; cursor:pointer;
    background:var(--ink); color:#fff; border:none; border-radius:14px;
    padding:11px 20px; font-size:14px;
    box-shadow:0 4px 0 #1c2b48; transition:transform .08s, box-shadow .08s;
  }
  .btn:active{transform:translateY(3px); box-shadow:0 1px 0 #1c2b48;}

  .warn{
    background:#FBE6C9; border:2px dashed var(--gold); color:#8a5a16;
    border-radius:12px; padding:9px 14px; font-size:13px; font-weight:700;
    margin-bottom:16px;
  }

  /* opponents row */
  .opponents{
    display:grid; gap:14px; margin-bottom:16px;
    grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  }
  .seat{
    background:var(--card); border:2px solid var(--line); border-radius:18px;
    padding:14px; box-shadow:0 6px 0 var(--line);
  }
  .seat.you{border-color:var(--coral); box-shadow:0 6px 0 #f0b6ac;}
  .seat-head{
    display:flex; justify-content:space-between; align-items:baseline;
    margin-bottom:10px;
  }
  .seat-name{font-family:'Mochiy Pop One',sans-serif; font-size:16px;}
  .seat.you .seat-name{color:var(--coral);}
  .seat-score{
    font-family:'Mochiy Pop One',sans-serif; font-size:22px; color:var(--ink);
  }
  .tableau{display:flex; flex-wrap:wrap; gap:6px; min-height:46px;}
  .empty-note{color:var(--ink-soft); font-size:12px; font-style:italic; padding:6px 0;}

  /* a collected-card chip */
  .chip{
    display:flex; align-items:center; gap:4px;
    background:var(--paper-deep); border:2px solid var(--line);
    border-radius:10px; padding:3px 8px 3px 6px; font-size:13px; font-weight:700;
  }
  .chip .em{font-size:16px;}
  .chip .ct{
    background:var(--ink); color:#fff; border-radius:7px;
    padding:0 6px; font-size:11px; font-family:'Mochiy Pop One',sans-serif;
  }
  .chip.pudding{background:#EFE3D2; border-color:#D8C3A5;}
  .chip.wasabi-open{border-color:var(--sage); border-style:dashed;}

  .last-draft{
    margin-top:9px; font-size:12px; color:var(--ink-soft); font-weight:700;
    min-height:18px;
  }
  .last-draft .em{font-size:14px;}

  /* your hand */
  .handbox{
    background:var(--card); border:2px solid var(--coral); border-radius:20px;
    padding:16px; box-shadow:0 7px 0 #f0b6ac;
  }
  .hand-title{
    font-family:'Mochiy Pop One',sans-serif; font-size:15px; color:var(--coral);
    margin-bottom:12px; display:flex; align-items:center; gap:8px;
  }
  .hand-title .dot{
    width:9px; height:9px; border-radius:50%; background:var(--coral);
    animation:pulse 1.4s infinite;
  }
  @keyframes pulse{0%,100%{opacity:1;}50%{opacity:.25;}}
  .hand{display:flex; flex-wrap:wrap; gap:10px;}

  /* a playable hand card */
  .card{
    width:84px; cursor:pointer; border:none; background:none; padding:0;
    font-family:'Zen Maru Gothic',sans-serif;
    transition:transform .12s ease;
  }
  .card:hover{transform:translateY(-8px) rotate(-2deg);}
  .card:disabled{cursor:default; opacity:.55;}
  .card:disabled:hover{transform:none;}
  .card-inner{
    background:var(--paper); border:3px solid var(--ink); border-radius:14px;
    padding:10px 6px 8px; text-align:center; box-shadow:0 5px 0 var(--ink);
    position:relative;
  }
  .card:active .card-inner{box-shadow:0 1px 0 var(--ink); transform:translateY(4px);}
  .card-em{font-size:34px; line-height:1; display:block;}
  .card-label{
    font-size:12px; font-weight:700; margin-top:6px;
    font-family:'Mochiy Pop One',sans-serif;
  }
  .roll-badge{
    position:absolute; top:-9px; right:-9px; background:var(--sage); color:#fff;
    width:24px; height:24px; border-radius:50%; border:2px solid #fff;
    font-size:12px; font-family:'Mochiy Pop One',sans-serif;
    display:flex; align-items:center; justify-content:center;
  }

  /* round banner toast */
  .toast{
    position:fixed; top:18px; left:50%; transform:translateX(-50%) translateY(-120px);
    background:var(--ink); color:#fff; padding:12px 26px; border-radius:16px;
    font-family:'Mochiy Pop One',sans-serif; font-size:16px;
    box-shadow:0 8px 24px var(--shadow); transition:transform .4s ease; z-index:30;
  }
  .toast.show{transform:translateX(-50%) translateY(0);}

  /* game-over modal */
  .modal-bg{
    position:fixed; inset:0; background:rgba(42,61,102,.55);
    display:none; align-items:center; justify-content:center; z-index:40; padding:20px;
  }
  .modal-bg.show{display:flex;}
  .modal{
    background:var(--paper); border:4px solid var(--ink); border-radius:24px;
    padding:30px 34px; max-width:440px; text-align:center;
    box-shadow:0 14px 0 var(--ink);
  }
  .modal h2{
    font-family:'Mochiy Pop One',sans-serif; font-size:26px; margin-bottom:6px;
  }
  .modal .sub{color:var(--ink-soft); font-weight:700; margin-bottom:18px;}
  .final-row{
    display:flex; justify-content:space-between; padding:8px 12px;
    border-radius:10px; margin:5px 0; font-weight:700; background:var(--card);
    border:2px solid var(--line);
  }
  .final-row.win{border-color:var(--coral); background:#FBE6C9;}
  .final-row .pts{font-family:'Mochiy Pop One',sans-serif;}
  footer{
    text-align:center; margin-top:26px; color:var(--ink-soft);
    font-size:12px; font-weight:700;
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Sushi <span class="go">Go!</span> &nbsp;vs&nbsp; DQN</h1>
    <div class="badges">
      <div class="badge">Round <b id="b-round">1</b>/3</div>
      <div class="badge">Turn <b id="b-turn">0</b></div>
      <button class="btn" id="new-game">New Game</button>
    </div>
  </header>

  <div class="warn" id="warn" style="display:none">
    &#9888; Model not loaded &mdash; opponents are playing <u>randomly</u>.
  </div>

  <div class="opponents" id="opponents"></div>

  <div class="handbox">
    <div class="hand-title"><span class="dot"></span> Your hand &mdash; pick a card to draft</div>
    <div class="hand" id="hand"></div>
  </div>

  <footer>You are seat 0 &middot; your trained Q-network controls the other seats &middot;
    all players draft simultaneously each turn</footer>
</div>

<div class="toast" id="toast"></div>

<div class="modal-bg" id="modal-bg">
  <div class="modal">
    <h2 id="modal-title">Game Over</h2>
    <div class="sub" id="modal-sub"></div>
    <div id="modal-rows"></div>
    <button class="btn" id="modal-new" style="margin-top:18px">Play Again</button>
  </div>
</div>

<script>
let META = [];
let prevRound = 1;
let busy = false;

async function api(path, body){
  const opt = {method: body ? 'POST':'GET'};
  if(body){ opt.headers={'Content-Type':'application/json'}; opt.body=JSON.stringify(body); }
  const r = await fetch(path, opt);
  return r.json();
}

function chip(cardType, count, extra){
  const m = META[cardType];
  const roll = m.rolls ? ` ${m.rolls}\u00D7\ud83c\udf00` : '';
  return `<div class="chip ${extra||''}">
            <span class="em">${m.emoji}</span>${m.label}${roll}
            <span class="ct">${count}</span>
          </div>`;
}

function renderTableau(p){
  let html = '';
  for(let t=0; t<p.tableau.length; t++){
    if(p.tableau[t] > 0) html += chip(t, p.tableau[t]);
  }
  if(p.pudding > 0) html += chip(10, p.pudding, 'pudding');
  if(p.wasabi_open > 0)
    html += `<div class="chip wasabi-open"><span class="em">\u23F3</span>
             wasabi ready <span class="ct">${p.wasabi_open}</span></div>`;
  if(!html) html = '<div class="empty-note">no cards yet this round</div>';
  return html;
}

function render(s){
  META = s.card_meta;
  document.getElementById('b-round').textContent = s.round;
  document.getElementById('b-turn').textContent  = s.turn;
  document.getElementById('warn').style.display  = s.model_ok ? 'none':'block';

  // opponents + you
  const box = document.getElementById('opponents');
  box.innerHTML = '';
  s.players.forEach(p=>{
    let draftHtml = '';
    if(p.last_draft !== null && p.last_draft !== undefined){
      const m = META[p.last_draft];
      draftHtml = `just drafted <span class="em">${m.emoji}</span> ${m.label}`;
    }
    box.innerHTML += `
      <div class="seat ${p.is_human?'you':''}">
        <div class="seat-head">
          <span class="seat-name">${p.name}</span>
          <span class="seat-score">${p.score}</span>
        </div>
        <div class="tableau">${renderTableau(p)}</div>
        <div class="last-draft">${draftHtml}</div>
      </div>`;
  });

  // your hand
  const hand = document.getElementById('hand');
  hand.innerHTML = '';
  if(s.done){
    hand.innerHTML = '<div class="empty-note">game over &mdash; start a new game</div>';
  } else {
    s.hand.forEach(ct=>{
      const m = META[ct];
      const badge = m.rolls ? `<span class="roll-badge">${m.rolls}</span>`:'';
      const c = document.createElement('button');
      c.className = 'card';
      c.innerHTML = `<div class="card-inner">${badge}
                       <span class="card-em">${m.emoji}</span>
                       <span class="card-label">${m.label}</span>
                     </div>`;
      c.onclick = ()=>playCard(ct);
      hand.appendChild(c);
    });
  }

  // round-change toast
  if(s.round !== prevRound && !s.done){
    showToast(`Round ${s.round} \u2014 fresh hands!`);
    prevRound = s.round;
  }

  if(s.done) showGameOver(s);
}

function showToast(msg){
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 2200);
}

function showGameOver(s){
  const winners = s.winners;
  const youWon = winners.includes(0);
  document.getElementById('modal-title').textContent =
    youWon ? '\ud83c\udf89 You Win!' : 'Game Over';
  let sub;
  if(winners.length > 1) sub = 'It\u2019s a tie!';
  else sub = winners.includes(0) ? 'You beat the DQN.' :
             `${s.players[winners[0]].name} wins.`;
  document.getElementById('modal-sub').textContent = sub;

  const rows = document.getElementById('modal-rows');
  rows.innerHTML = '';
  const order = [...s.players.keys()].sort((a,b)=>s.players[b].score-s.players[a].score);
  order.forEach(i=>{
    const p = s.players[i];
    rows.innerHTML += `<div class="final-row ${winners.includes(i)?'win':''}">
        <span>${p.name} &middot; \ud83c\udf6e${p.pudding}</span>
        <span class="pts">${p.score}</span></div>`;
  });
  document.getElementById('modal-bg').classList.add('show');
}

async function playCard(ct){
  if(busy) return;
  busy = true;
  const s = await api('/api/move', {card: ct});
  render(s);
  busy = false;
}

async function newGame(){
  document.getElementById('modal-bg').classList.remove('show');
  prevRound = 1;
  const s = await api('/api/new_game', {});
  render(s);
}

document.getElementById('new-game').onclick = newGame;
document.getElementById('modal-new').onclick = newGame;

// boot
api('/api/state').then(render);
</script>
</body>
</html>
"""


# ======================================================================================
# Entry point
# ======================================================================================
def main():
    global SESSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sushi_go_qnet.pt",
                        help="path to the trained DQN checkpoint")
    parser.add_argument("--n-players", type=int, default=3, choices=[2, 3, 4],
                        help="MUST match the player count the model was trained on")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    obs_dim = SushiGoParallelEnv(n_players=args.n_players).obs_dim
    model = load_model(args.n_players, obs_dim, args.model)
    SESSION = GameSession(args.n_players, model)

    print(f"\n  Sushi Go vs DQN  —  open  http://127.0.0.1:{args.port}  in your browser\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()