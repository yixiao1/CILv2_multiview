#!/bin/bash

# Collect data with the Roach RL agent

data_collection_valid_busy_roachRL_Town02_clearnoon () {
    python ${DRIVING_TEST_ROOT}/driving/evaluator.py \
    --debug=0 \
    --scenarios=${DRIVING_TEST_ROOT}/data/data_collection/valid_Town02_busy_clearnoon.json  \
    --routes=${DRIVING_TEST_ROOT}/data/data_collection \
    --repetitions=1 \
    --resume=True \
    --track=SENSORS \
    --agent=${DRIVING_TEST_ROOT}/driving/autoagents/RoachRL_expert.py \
    --checkpoint=${DRIVING_TEST_ROOT}/results/data_collection  \
    --agent-config=${TRAINING_RESULTS_ROOT}/_results/Roach_rl_birdview/config11833344.json \
    --docker=carlasim/carla:0.9.13 \
    --gpus=5 \
    --fps=10 \
    --PedestriansSeed=0 \
    --trafficManagerSeed=0 \
    --data-collection
}

function_array=("data_collection_valid_busy_roachRL_Town02_clearnoon")


# resume benchmark in case carla is crashed, until the benchmark is finished
RED=$'\e[0;31m'
NC=$'\e[0m'
for run in "${function_array[@]}"; do
    PYTHON_RETURN=1
    until [ $PYTHON_RETURN == 0 ]; do
      ${run}
      PYTHON_RETURN=$?
      echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
      sleep 2
    done
    sleep 2
done

echo "Bash script done."