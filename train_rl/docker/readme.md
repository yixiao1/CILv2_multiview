## Build the image

    docker image build  -t danielc11/cil:0.0 --build-arg WANDB_API_KEY=$WANDB_API_KEY .

    docker push danielc11/cil:0.0

    docker pull danielc11/cil:0.0

## Spawn the container - interactive

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/danc/results/rlad2:/root/results/rlad2 -v /home/danc/rlad2:/root/rlad2 danielc11/rlad2:0.2 bash

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/danc/PhD/autonomous_driving/CARLA_0.9.13:/root/CARLA_0.9.13 -v /home/danc/PhD/autonomous_driving/CILv2_multiview:/root/CILv2_multiview danielc11/cil:0.0  bash

    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --network=host -v /home/danc/PhD/autonomous_driving/CARLA_0.9.13:/root/CARLA_0.9.13 -v /home/danc/PhD/autonomous_driving/CILv2_multiview:/root/CILv2_multiview -v /home/danc/results/CILv2_multiview:/root/results/CILv2_multiview danielc11/cil:0.0  bash

## Attach to a running container

    docker attach <container_id>

## Dettach from running container

    press CRTL-p & CRTL-q

## wandb debugging:

    WANDB_MODE=disabled python3 main.py -en experiment_name1 -ow