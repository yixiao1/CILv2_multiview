import os
import torch
import torch.distributed as dist
import time
import shutil
import numpy as np
from configs import g_conf, set_type_of_process, merge_with_yaml
from network.models_console import Models
from _utils.training_utils import seed_everything, DataParallelWrapper, DataParallelDPPWrapper, check_saved_checkpoints, update_learning_rate
from _utils.utils import extract_targets, extract_other_inputs, extract_commands, print_train_info, test_stop
from _utils.evaluation import evaluation_saving
from logger import _logger


def update_early_stopping(flags, rank, world_size):
    torch.cuda.set_device(rank)
    data_dpp = {
        'flags': flags,
        'rank': rank
    }
    outputs_dpp = [None for _ in range(world_size)]
    dist.all_gather_object(outputs_dpp, data_dpp)    # we only want to operate on the collected objects at master node

    flags = [el['flags'] for el in outputs_dpp if el['rank'] == 0][0]


def train_upstream_task(model, optimizer, rank=0, world_size=1):
    """
    Upstream task is for training your model

    """
    early_stopping_flags = []
    acc_time = 0.0
    time_start = time.time()
    local_iteration = 0
    init_iteration = model._current_iteration
    init_epoch = (model._current_iteration * g_conf.BATCH_SIZE // len(model))

    while True:
        # we get dataloader of the model
        dataloader = model._train_loader
        if world_size > 1:
            dataloader.sampler.set_epoch(init_epoch)

        for data in dataloader:
            if rank == 0:
                early_stopping_flags = evaluation_saving(model, optimizer, early_stopping_flags, save_all_checkpoints=True)
            if world_size > 1:
                flags = update_early_stopping(early_stopping_flags, rank, world_size)

            if early_stopping_flags and all(early_stopping_flags[-int(g_conf.EARLY_STOPPING_PATIENCE):]):
                print(' Apply early stopping, training stopped !')
                break

            if g_conf.LEARNING_RATE_DECAY:
                if model._done_epoch in g_conf.LEARNING_RATE_DECAY_EPOCHES and ((model._current_iteration-1)*g_conf.BATCH_SIZE <= len(model) * model._done_epoch):
                    update_learning_rate(optimizer, minimumlr=g_conf.LEARNING_RATE_MINIMUM)

            if world_size > 1:
                src_images = [[data['current'][i][camera_type].to(f'cuda:{model.device_ids[0]}') for camera_type in g_conf.DATA_USED] for i in range(len(data['current']))]
                src_directions = [extract_commands(data['current'][i]['can_bus']['direction']).to(f'cuda:{model.device_ids[0]}') for i in
                                  range(len(data['current']))]
                src_s = [extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                         ignore=['direction']).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['future']))]
                else:
                    tgt_a = [extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
            else:
                src_images = [[data['current'][i][camera_type].cuda() for camera_type in g_conf.DATA_USED] for i in range(len(data['current']))]
                src_directions = [extract_commands(data['current'][i]['can_bus']['direction']).cuda() for i in
                                  range(len(data['current']))]
                src_s = [extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                         ignore=['direction']).cuda() for i in range(len(data['current']))]
                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).cuda() for i in range(len(data['future']))]
                else:
                    tgt_a = [extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).cuda() for i in range(len(data['current']))]


            # src_images = src_images.to(f'cuda:{model.device_ids[0]}')
            # src_directions = src_directions.to(f'cuda:{model.device_ids[0]}')
            # src_s = src_s.to(f'cuda:{model.device_ids[0]}')
            # model.to(f'cuda:{model.device_ids[0]}')
            if g_conf.USE_AUTOCAST:
                with torch.cuda.amp.autocast():
                    action_outputs = model.forward(src_images, src_directions, src_s)

                    loss_params = {
                        'action_output': action_outputs,
                        'targets_action': tgt_a,
                        'variable_weights': g_conf.LOSS_WEIGHT
                    }

                    if g_conf.ACCELERATION_AS_ACTION:
                        loss, steer_loss, acceleration_loss = model.loss(loss_params)
                        if rank == 0:
                            acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                        acc_time, loss.item(), steer_loss.item(), acceleration_loss.item())
                    else:
                        loss, steer_loss, throttle_loss, brake_loss = model.loss(loss_params)
                        if rank == 0:
                            acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                        acc_time, loss.item(), steer_loss.item(), throttle_loss.item(), brake_loss.item)
            else:
                action_outputs = model.forward(src_images, src_directions, src_s)
                loss_params = {
                    'action_output': action_outputs,
                    'targets_action': tgt_a,
                    'variable_weights': g_conf.LOSS_WEIGHT
                }

                if g_conf.ACCELERATION_AS_ACTION:
                    loss, steer_loss, acceleration_loss = model.loss(loss_params)
                    if rank == 0:
                        acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss.item(), steer_loss.item(), acceleration_loss.item())
                else:
                    loss, steer_loss, throttle_loss, brake_loss = model.loss(loss_params)
                    if rank == 0:
                        acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss.item(), steer_loss.item(), throttle_loss.item(), brake_loss.item)

            time_start = time.time()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """
            ################################################
                Adding tensorboard logs
            #################################################
            """
            if rank == 0:
                _logger.add_scalar('Loss', loss.item(), model._current_iteration)

                ## Adding loss to tensorboard
                _logger.add_scalar('Loss_steer', steer_loss.item(), model._current_iteration)
                if g_conf.ACCELERATION_AS_ACTION:
                    _logger.add_scalar('Loss_acceleration', acceleration_loss.item(), model._current_iteration)
                else:
                    _logger.add_scalar('Loss_throttle', throttle_loss.item(), model._current_iteration)
                    _logger.add_scalar('Loss_brake', brake_loss.item(), model._current_iteration)

            if test_stop(g_conf.NUMBER_EPOCH * len(model), model._current_iteration * g_conf.BATCH_SIZE):
                print('')
                print('Training finished !!')
                break

            local_iteration += 1

            model._current_iteration = init_iteration + local_iteration
            model._done_epoch = (model._current_iteration * g_conf.BATCH_SIZE // len(model))

            if world_size > 1:
                dataloader.sampler.set_epoch(model._done_epoch - 1)

            del src_images
            del src_directions
            del tgt_a
            del src_s
            del action_outputs
        else:
            continue
        break


# The main function maybe we could call it with a default name
def execute(gpus_list, exp_batch, exp_name, rank=0):
    """
        The main training function for decoder.
    Args:
        gpus_list: The list of all GPU can be used
        exp_batch: The folder with the experiments
        exp_name: the alias, experiment name

    Returns:
        None

    """
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    print(torch.cuda.device_count(), 'GPUs to be used: ', gpus_list)
    merge_with_yaml(os.path.join('configs', exp_batch, exp_name + '.yaml'))
    shutil.copyfile(os.path.join('configs', exp_batch, exp_name + '.yaml'),
                    os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results',
                                 g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME, exp_name + '.yaml'))
    set_type_of_process('train_val', root=os.environ["TRAINING_RESULTS_ROOT"], rank=rank)
    seed_everything(seed=g_conf.MAGICAL_SEED)

    gpus_list_int = range(len(gpus_list))

    # gpus_list_str = ",".join(gpus_list)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus_list_str
    # model = DataParallelWrapper(model, device_ids=gpus_list_int)

    if len(gpus_list) > 1:
        device_id = rank % torch.cuda.device_count()
        device_id = gpus_list_int[device_id]
        g_conf.MODEL_CONFIGURATION['rank'] = rank
        g_conf.MODEL_CONFIGURATION['num_process'] = len(gpus_list)
    else:
        g_conf.MODEL_CONFIGURATION['rank'] = 0
        g_conf.MODEL_CONFIGURATION['num_process'] = 1

    model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    # print("===================== Model Configuration =====================")
    # print("")
    # print(model)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('model params: ', num_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=g_conf.LEARNING_RATE)
    if len(gpus_list) > 1 and g_conf.DATA_PARALLEL:
        print("Using multiple GPUs parallel! ")
        # model = DataParallelWrapper(model)
        # gpus_list_int = [int(el) for el in gpus_list]
        model.to(device_id)
        model = DataParallelDPPWrapper(model, device_ids=[device_id], find_unused_parameters=True)

    # To load a specific checkpoint
    if g_conf.LOAD_CHECKPOINT:
        latest_checkpoint = os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,
                                                                g_conf.EXPERIMENT_NAME, 'checkpoints', g_conf.LOAD_CHECKPOINT)

    # To train model from scratch, or to resume training on a previous one
    elif g_conf.TRAINING_RESUME:
        latest_checkpoint = check_saved_checkpoints(os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,
                                                                g_conf.EXPERIMENT_NAME, 'checkpoints'))
    elif g_conf.FINETUNE:
        latest_checkpoint = None
        finetune_checkpoint = torch.load(g_conf.FINETUNE_MODEL)
        pretrained_dict = finetune_checkpoint['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)

        print('')
        print('    Finetunning model from -> ', g_conf.FINETUNE_MODEL)
    else:
        latest_checkpoint = None


    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint)
        pretrained_dict = checkpoint['model']

        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # we manually move optimizer state to GPU memory after loading it from the checkpoint
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v):
                    if len(gpus_list) > 1:
                        state[k]=v.to(f'cuda:{model.device_ids[0]}')
                    else:
                        state[k]=v.cuda()
        for param_group in optimizer.param_groups:
            print('')
            print('    Resum training from epoch -> ', checkpoint['epoch'])
            print('    Resum the latest learning rate -> ', param_group['lr'])
            if g_conf.LEARNING_RATE_DECAY:
                print('      - learning rate decay at epoch', g_conf.LEARNING_RATE_DECAY_EPOCHES, ', minimum lr:', g_conf.LEARNING_RATE_MINIMUM)
            print('')
            print('=======================================================================================')
            print('')

        model._current_iteration = checkpoint['iteration'] + 1
        model._done_epoch = checkpoint['epoch']
    else:
        print('')
        print('    Training from scratch')
        print('    Initial learning rate -> ', g_conf.LEARNING_RATE)
        if g_conf.LEARNING_RATE_DECAY:
            print('      - learning rate decay at epoch', g_conf.LEARNING_RATE_DECAY_EPOCHES, ', minimum lr:', g_conf.LEARNING_RATE_MINIMUM)
        print('')
        print('=======================================================================================')
        print('')

    if len(gpus_list) > 1:
        # model.to(f'cuda:{model.device_ids[0]}')
        model.to(device_id)
        # optimizer.to(f'cuda:{model.device_ids[0]}')
    else:
        model.cuda()
        # optimizer.cuda()
    model.train()
    train_upstream_task(model, optimizer, rank=rank, world_size=len(gpus_list))

