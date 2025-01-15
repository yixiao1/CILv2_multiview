#!/bin/bash

# Collect data with the Roach RL agent

ted_data_collection_dense_roachRLplus_Town01_hardrainnoon () {
    python ${DRIVING_TEST_ROOT}/driving/evaluator.py \
    --debug=0 \
    --scenarios=${DRIVING_TEST_ROOT}/data/data_collection/ted_training_Town01_dense_hardrainnoon.json  \
    --routes=${DRIVING_TEST_ROOT}/data/data_collection \
    --repetitions=1 \
    --resume=True \
    --track=SENSORS \
    --agent=${DRIVING_TEST_ROOT}/driving/autoagents/RoachRL_expert.py \
    --checkpoint=${DRIVING_TEST_ROOT}/results/ted_data_collection  \
    --agent-config=${TRAINING_RESULTS_ROOT}/_results/Roach_rl_birdview/config11833344.json \
    --docker=carlasim/carla:0.9.14 \
    --gpus=2 \
    --fps=10 \
    --PedestriansSeed=2 \
    --trafficManagerSeed=2 \
    --data-collection
}

function_array=("ted_data_collection_dense_roachRLplus_Town01_hardrainnoon")


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