#!/bin/bash

# * To run nocrash benchmark for trained agent on an empty town

nocrash_newweathertown_empty () {
    python3 ${DRIVING_TEST_ROOT}/driving/evaluator.py \
    --debug=0 \
    --scenarios=${DRIVING_TEST_ROOT}/data/nocrash/nocrash_newweathertown_empty_Town02_lbc.json  \
    --routes=${DRIVING_TEST_ROOT}/data/nocrash \
    --repetitions=1 \
    --resume=True \
    --track=SENSORS \
    --agent=${DRIVING_TEST_ROOT}/driving/autoagents/CILv2_agent.py \
    --checkpoint=${DRIVING_TEST_ROOT}/results/nocrash  \
    --agent-config=${TRAINING_RESULTS_ROOT}/_results/CIL_ViT_oneseq/CILv2_3cam_vitb32_Town01Full_oneseq_bs256_fPE_wmupcdown_lr1e4_preACT_noCLS_out1MLP_hres_CMDSPD_sensorEmb/config60.json \
    --docker=carlasim/carla:0.9.13 \
    --gpus=2 \
    --fps=20 \
    --PedestriansSeed=0 \
    --trafficManagerSeed=0 \
    --save-driving-vision
}

function_array=("nocrash_newweathertown_empty")


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