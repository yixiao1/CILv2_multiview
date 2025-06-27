#!/bin/bash

# Run NoCrash benchmark for trained agent in Town02 for SoftRainSunset and WetSunset weathers

# Check if at least five arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 GPU_ID (int) EXPERIMENT_FOLDER (str) EXPERIMENT_NAME (str) EPOCH (int) SCENARIO {empty, regular, busy} 
    [--random-seed (optional, int)] [--save-driving-vision (optional, bool)] [--fps (optional, int)] [--agent (optional, str: TED_agent, CILv2_agent)]"
    exit 1
fi

# Assign the arguments to variables
GPU_ID=$1
EXPERIMENT_FOLDER=$2
EXPERIMENT_NAME=$3
EPOCH=$4
SCENARIO=$5

shift 5  # Remove the first five arguments

RANDOM_SEED=0  # Default; change when running multiple times to test for variance
SAVE_DRIVING_VISION=false
FPS=20  # Default frames per second for the evaluation in synchronous mode
AGENT="TED_agent"
DOCKER=carlasim/carla:0.9.13

# Process optional arguments
while (( "$#" )); do
  case "$1" in
    --save-driving-vision)
      SAVE_DRIVING_VISION=true
      shift
      ;;
    --random-seed)
      shift
      RANDOM_SEED=$1
      shift
      ;;
    --fps)
      shift
      FPS=$1
      shift
      ;;
    --agent)
      shift
      AGENT=$1
      shift
      ;;
    --docker)
      shift
      DOCKER=$1
      shift
      ;;
    *)
      echo "Error: Invalid argument $1"
      exit 1
      ;;
  esac
done

# Format EPOCH to have at least two digits (pad with a leading zero if necessary)
EPOCH=$(printf "%02d" "$EPOCH")
# Construct the path to the agent and scenario config files
AGENT_CONFIG=$TRAINING_RESULTS_ROOT/_results/$EXPERIMENT_FOLDER/$EXPERIMENT_NAME/config"$EPOCH".json
SCENARIO_FILE=$DRIVING_TEST_ROOT/data/ted/nocrash_newtown_"$SCENARIO"_Town02_lbc.json

# Check if the scenario file exists
if [ ! -f "$SCENARIO_FILE" ]; then
    echo "Invalid scenario type or file does not exist. Expected 'empty', 'regular', or 'busy'."
    exit 1
fi

python_command="python3 $DRIVING_TEST_ROOT/driving/evaluator.py \
    --debug=0 \
    --scenarios=$SCENARIO_FILE  \
    --routes=$DRIVING_TEST_ROOT/data/ted \
    --repetitions=1 \
    --resume=True \
    --track=SENSORS \
    --agent=$DRIVING_TEST_ROOT/driving/autoagents/$AGENT.py \
    --checkpoint=$DRIVING_TEST_ROOT/results/ted  \
    --agent-config=$AGENT_CONFIG \
    --docker=$DOCKER \
    --gpus=$GPU_ID \
    --fps=$FPS \
    --PedestriansSeed=$RANDOM_SEED \
    --trafficManagerSeed=$RANDOM_SEED"

# Add the --save-driving-vision flag only for the "busy" scenario
if [ "$SAVE_DRIVING_VISION" = "true" ]; then
    echo "Saving driving vision! Results will be found in $DATASET_PATH/driving_record/nocrash_newtown_"$SCENARIO"_Town02_lbc..."
    python_command+=" --save-driving-vision"
fi

# resume benchmark in case carla is crashed, until the benchmark is finished
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  $python_command
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done
sleep 2

echo "Bash script done."