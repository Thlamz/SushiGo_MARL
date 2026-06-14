# SushiGo_MARL
Requirement for the Multi Agent Systems class

## Requirements 

- Docker

- Nvidia toolkit container

## Environment

Build the image:

    docker build -f dockerfile -t sushi_go .

Create the container:

    docker-compose up -d

    docker exec -it sushi_go_container bash

## Trainning the DQN 

Just run:

    python3 train/train_dqn.py --save-path models/DQN/NAME_OF_THE_MODEL.pt

## Testing

Just run:

    python3 test/play.py --model PATH_TO_YOUR_MODEL --n-players NUMBER_OF_PLAYER