# Player-Count Generalizable Results

This folder contains results for experiments focused on **player-count generalization** in Sushi Go multi-agent reinforcement learning. The goal is to train models that perform well across different table sizes, especially 2-, 3-, and 4-player games, instead of specializing to a single fixed number of opponents.

In this setting, a player-count generalizable policy should learn game-relevant decision patterns that remain useful when the number of other agents changes. This matters because Sushi Go dynamics shift with table size: card availability, competition for scoring combinations, denial opportunities, and expected future hands all depend on how many players are participating.

## Strategy

Our approach combines three ideas:

1. **Padded / sequential observation structure**

   Observations are structured so the model can consume a consistent input shape even when the number of players changes. Player-specific information can be represented sequentially, with padding used for missing players in smaller games. This allows a single model interface to support 2-, 3-, and 4-player environments.

2. **Stochastic agent count during training**

   Instead of training only on one table size, the number of agents is sampled during training. By exposing the policy to multiple player counts, the model is encouraged to learn behavior that transfers across table sizes rather than memorizing patterns specific to one configuration.

3. **Optional encoder architecture**

   The encoder-based model is included as a stronger architectural option for learning compact representations from the padded/sequential observations. The encoder can help the policy separate meaningful player-state information from padding and support better generalization across variable numbers of agents.