import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

import math

import time
import shutil
import resource
from typing import List
from configs import g_conf, set_type_of_process, merge_with_yaml
from network.models_console import Models
from _utils.training_utils import seed_everything, DataParallelDPPWrapper, check_saved_checkpoints, update_learning_rate
from _utils import utils
from _utils.evaluation import evaluation_saving
from logger import _logger, StdoutLogger
from einops import reduce


def update_early_stopping(flags, rank, world_size):
    torch.cuda.set_device(rank)
    data_dpp = {
        'flags': flags,
        'rank': rank
    }
    outputs_dpp = [None for _ in range(world_size)]
    dist.all_gather_object(outputs_dpp, data_dpp)    # we only want to operate on the collected objects at master node
    flags = []
    for el in outputs_dpp:
        if el is not None:
            flags.append(el['flags']) if el['rank'] == 0 else None
    flags = flags[0]


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

    total_iterations = g_conf.NUMBER_EPOCH * len(model) // g_conf.BATCH_SIZE

    if g_conf.AUTOCAST:
        # Scale the gradients since we use autocast
        scaler = torch.cuda.amp.GradScaler()

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

            # Update learning rate according to selected schedule
            if g_conf.LEARNING_RATE_DECAY:
                                # Step learning rate decay
                if g_conf.LEARNING_RATE_SCHEDULE == 'step':
                    # Update only at the specific schedule
                    if model._done_epoch in g_conf.LEARNING_RATE_DECAY_EPOCHES and \
                            ((model._current_iteration-1)*g_conf.BATCH_SIZE <= len(model) * model._done_epoch):
                        update_learning_rate(optimizer)
                elif g_conf.LEARNING_RATE_SCHEDULE == 'warmup_cooldown':
                    # Update at each iteration
                    update_learning_rate(optimizer, iteration=model._current_iteration - 1,
                                         total_iterations=total_iterations)

            if world_size > 1:
                # Here: add attention masks
                src_images = [[data['current'][i][camera_type].to(f'cuda:{model.device_ids[0]}')
                               for camera_type in g_conf.DATA_USED if 'rgb' in camera_type]
                              for i in range(len(data['current']))]
                src_directions = [
                    utils.extract_commands(data['current'][i]['can_bus']['direction']).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
                src_s = [utils.extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                                    ignore=['direction']).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
                if g_conf.ATTENTION_AS_INPUT:
                    if not g_conf.ATTENTION_FROM_UNET:
                        src_attn_masks = [[data['current'][i][camera_type].to(f'cuda:{model.device_ids[0]}')
                         for camera_type in g_conf.DATA_USED if 'virtual_attention' in camera_type]
                            for i in range(len(data['current']))]
                    else:
                        src_attn_masks = None  # TODO: comes from UNet prediction!

                    if g_conf.ATTENTION_AS_NEW_CHANNEL:
                        # Add the masks as the alpha channel in the src_images
                        for i in range(len(src_images)):
                            for j in range(len(src_images[i])):
                                src_images[i][j] = torch.cat((src_images[i][j], src_attn_masks[i][j]), 1)
                    else:
                        # Element-wise multiplication of the masks with the src_images
                        for i in range(len(src_images)):
                            for j in range(len(src_images[i])):
                                src_images[i][j] = src_images[i][j] * src_attn_masks[i][j]

                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [utils.extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['future']))]
                else:
                    tgt_a = [utils.extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]

                if g_conf.ATTENTION_LOSS:
                    src_atts_left = [value.to(f'cuda:{model.device_ids[0]}') for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_left_' in key]
                    src_atts_central = [value.to(f'cuda:{model.device_ids[0]}') for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_central_' in key]
                    src_atts_right = [value.to(f'cuda:{model.device_ids[0]}') for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_right_' in key]

                    if 'Attention_KL' in g_conf.LOSS:
                        tgt_att = utils.prepare_target_attentions(src_atts_left[0], src_atts_central[0], src_atts_right[0], binarize=g_conf.BINARIZE_ATTENTION)
                    elif g_conf.LOSS == 'Action_nospeed_L1_Attention_L2':
                        tgt_att = torch.cat((src_atts_left[0], src_atts_central[0], src_atts_right[0]), 1)
                    else:
                        raise ValueError(f'Error! We cannot use the {g_conf.LOSS} loss with attention!')
            
            else:
                # Here: add attention masks
                src_images = [[data['current'][i][camera_type].cuda() 
                               for camera_type in g_conf.DATA_USED if 'rgb' in camera_type] 
                               for i in range(len(data['current']))]
                src_directions = [utils.extract_commands(data['current'][i]['can_bus']['direction']).cuda() for i in
                                  range(len(data['current']))]
                src_s = [utils.extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                         ignore=['direction']).cuda() for i in range(len(data['current']))]
                if g_conf.ATTENTION_AS_INPUT:
                    if not g_conf.ATTENTION_FROM_UNET:
                        src_attn_masks = [[data['current'][i][camera_type].cuda()
                                           for camera_type in g_conf.DATA_USED if 'virtual_attention' in camera_type]
                                          for i in range(len(data['current']))]
                    else:
                        src_attn_masks = None  # TODO: comes from UNet prediction!

                    # Add the masks as the alpha channel in the src_images or element-wise multiplication
                    for i in range(len(src_images)):
                        for j in range(len(src_images[i])):
                            src_images[i][j] = torch.cat((src_images[i][j], src_attn_masks[i][j]), 1) \
                                if g_conf.ATTENTION_AS_NEW_CHANNEL else src_images[i][j] * src_attn_masks[i][j]

                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [utils.extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).cuda() for i in range(len(data['future']))]
                else:
                    tgt_a = [utils.extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).cuda() for i in range(len(data['current']))]

                if g_conf.ATTENTION_LOSS:
                    src_atts_left = [value.cuda() for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_left_' in key]
                    src_atts_central = [value.cuda() for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_central_' in key]
                    src_atts_right = [value.cuda() for current_data in data['current'] for key, value in current_data.items() if 'virtual_attention_right_' in key]

                    if 'Attention_KL' in g_conf.LOSS:
                        tgt_att = utils.prepare_target_attentions(src_atts_left[0], src_atts_central[0], src_atts_right[0], binarize=g_conf.BINARIZE_ATTENTION)
                    elif g_conf.LOSS == 'Action_nospeed_L1_Attention_L2':
                        tgt_att = torch.cat((src_atts_left[0], src_atts_central[0], src_atts_right[0]), 1)

            if g_conf.USE_AUTOCAST:
                with torch.cuda.amp.autocast():
                    # if g_conf.CMD_SPD_TOKENS and g_conf.PREDICT_CMD_SPD:
                    #     action_outputs, (cmd_out, spd_out) = model.forward(src_images, src_directions, src_s)
                    # else:
                    action_outputs = model.forward(src_images, src_directions, src_s)

                    loss_params = {
                        'action_output': action_outputs,
                        'targets_action': tgt_a,
                        'variable_weights': g_conf.LOSS_WEIGHT
                    }

                    if g_conf.ACCELERATION_AS_ACTION:
                        loss, steer_loss, acceleration_loss = model.loss(loss_params)
                        if rank == 0:
                            acc_time = utils.print_train_info(
                                g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                acc_time, loss.item(), steer_loss.item(), acceleration_loss.item())
                    else:
                        loss, steer_loss, throttle_loss, brake_loss = model.loss(loss_params)
                        if rank == 0:
                            acc_time = utils.print_train_info(
                                g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                acc_time, loss.item(), steer_loss.item(), throttle_loss.item(), brake_loss.item)
            else:
                # if g_conf.CMD_SPD_TOKENS and g_conf.PREDICT_CMD_SPD:
                #     action_outputs, (cmd_out, spd_out) = model.forward(src_images, src_directions, src_s)
                # else:
                action_outputs, resnet_inter, att_out = model.forward(src_images, src_directions, src_s)
                # if isinstance(action_outputs, tuple):
                #     action_output_patches, action_output_tokens = action_outputs
                loss_params = {
                    'action_output': action_outputs,
                    'targets_action': tgt_a,
                    'variable_weights': g_conf.LOSS_WEIGHT
                }

                if g_conf.ADAPTIVE_QUANTILE_REGRESSION:
                    difference = action_outputs[:, -1, :] - tgt_a[-1]
                    aqr_tau = 0.5 + 0.4 * difference[:, -1].sign().mean().item()  # TODO: EMA?
                    loss_params['variable_weights']['tau'] = aqr_tau
                
                elif g_conf.ADAPTIVE_QUANTILE_REGRESSION_SCHED:
                    if model._current_iteration < total_iterations * 0.1:
                        aqr_tau = 0.5
                    else:
                        alpha = 5 * math.pi / (9 * total_iterations)
                        beta = - alpha * total_iterations / 10
                        aqr_tau = 0.5 + 0.4 * math.sin(alpha * model._current_iteration + beta)

                if g_conf.ATTENTION_LOSS:
                    if g_conf.EARLY_ATTENTION:
                        # Sizes of tensors:
                        #  - block 0 (RN_ATTENTION_LAYER=0): [cam*B, 64, 40, 40]
                        #  - block 1 (RN_ATTENTION_LAYER=1): [cam*B, 64, 75, 75]
                        #  - block 2 (RN_ATTENTION_LAYER=2): [cam*B, 128, 38, 38]
                        #  - block 3 (RN_ATTENTION_LAYER=3): [cam*B, 256, 19, 19]
                        #  - block 4 (RN_ATTENTION_LAYER=4 or -1): [cam*B, 512, 10, 10]
                        resnet_inter = resnet_inter[g_conf.RN_ATTENTION_LAYER]  # [cam*B, C, H, W]
                        resnet_inter = reduce(resnet_inter, '(b cam) c h w -> b cam h w', reduction='mean', cam=len([c for c in g_conf.DATA_USED if 'attention' in c]))
                        loss_params.update({'attention_output': resnet_inter,
                                            'targets_attention': tgt_att})
                    else:
                        # Just the average of the attention map of the last layer of the Encoder
                        loss_params.update(
                            {'attention_output': att_out[g_conf.TFX_ENC_ATTENTION_LAYER].mean(dim=1),
                             'targets_attention': tgt_att
                             })

                if g_conf.ACCELERATION_AS_ACTION:
                    if g_conf.ATTENTION_LOSS:
                        loss, steer_loss, acceleration_loss, att_loss = model.loss(loss_params)
                    else:
                        loss, steer_loss, acceleration_loss = model.loss(loss_params)
                        att_loss = None
                    if rank == 0:
                        acc_time = utils.print_train_info(
                            g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                            acc_time, loss.item(), steer_loss.item(), acceleration_loss.item(), att_loss_data=att_loss)
                else:
                    loss, steer_loss, throttle_loss, brake_loss, att_loss = model.loss(loss_params)
                    if rank == 0:
                        acc_time = utils.print_train_info(
                            g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                            acc_time, loss.item(), steer_loss.item(), throttle_loss.item(), brake_loss.item, att_loss_data=att_loss)

            time_start = time.time()

            optimizer.zero_grad()
            loss.backward()
            # Clip the grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
                if att_loss is not None:
                    _logger.add_scalar('Loss_attention', att_loss.item(), model._current_iteration)

                 # Extra logging
                _logger.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], model._current_iteration)
                # Cosine similarity of the learned [ACC] and [STR] tokens, if being used
                if not g_conf.ONE_ACTION_TOKEN and not g_conf.NO_ACT_TOKENS:
                    try:
                        accel_token = getattr(model._model, 'camera_tfx_accel_token')
                        steer_token = getattr(model._model, 'camera_tfx_steer_token')
                    except AttributeError:
                        accel_token = getattr(model._model, 'tfx_accel_token')
                        steer_token = getattr(model._model, 'tfx_steer_token')
                    cos_sim = F.cosine_similarity(accel_token,  # [ACC], [1, 1, D]
                                                  steer_token,  # [STR], [1, 1, D]
                                                  dim=-1).item()  # [1, 1, D] -> [1, 1] -> float
                    _logger.add_scalar('Cosine sim [STR] and [ACC]', cos_sim, model._current_iteration)
                # Keep track of action ratio if it's learnable
                if g_conf.LEARNABLE_ACTION_RATIO:
                    _logger.add_scalar('Action ratio', model._model.action_ratio.item(), model._current_iteration)

                # Log steering difference w/ target
                action_difference = action_outputs[:, -1, :] - tgt_a[-1]
                _logger.add_histogram('sign(steer_out-tgt_steer)', 
                                      action_difference[:, 0].sign().cpu().detach().numpy(), 
                                      model._current_iteration)
                _logger.add_scalar('Average sign(steer_out-tgt_steer)',
                                   action_difference[:, 0].sign().mean().item(),
                                   model._current_iteration)

                _logger.add_histogram('steer_out-tgt_steer', 
                                      action_difference[:, 0].cpu().detach().numpy(), 
                                      model._current_iteration)

                # Log acceleration difference w/ target
                _logger.add_histogram('sign(acc_out-tgt_acc)', 
                                      action_difference[:, 1].sign().cpu().detach().numpy(), 
                                      model._current_iteration)
                _logger.add_scalar('Average sign(acc_out-tgt_acc)',
                                   action_difference[:, 1].sign().mean().item(),
                                   model._current_iteration)

                _logger.add_histogram('acc_out-tgt_acc', 
                                      action_difference[:, 1].cpu().detach().numpy(), 
                                      model._current_iteration)
                
                if g_conf.ADAPTIVE_QUANTILE_REGRESSION or g_conf.ADAPTIVE_QUANTILE_REGRESSION_SCHED:
                    _logger.add_scalar('AQR - tau', aqr_tau, model._current_iteration)

            if utils.test_stop(g_conf.NUMBER_EPOCH * len(model), model._current_iteration * g_conf.BATCH_SIZE):
                print('\nTraining finished !!')
                break

            local_iteration += 1

            model._current_iteration = init_iteration + local_iteration
            model._done_epoch = (model._current_iteration * g_conf.BATCH_SIZE // len(model))

            if world_size > 1:
                dataloader.sampler.set_epoch(model._done_epoch - 1)

            # Free the memory
            del src_images
            del src_directions
            del tgt_a
            del src_s
            del action_outputs
            if g_conf.ATTENTION_LOSS:
                del src_atts_left
                del src_atts_central
                del src_atts_right
                del tgt_att
        else:
            continue
        break


# The main function maybe we could call it with a default name
def execute(gpus_list: List[int], exp_batch: str, exp_name: str, rank: int = 0):
    """
        The main training function for decoder.
    Args:
        gpus_list: The list of all GPU can be used
        exp_batch: The folder with all the experiments
        exp_name: the alias, experiment name

    Returns:
        None

    """
    # Merge the yaml file with the default configuration
    merge_with_yaml(os.path.join('configs', exp_batch, exp_name + '.yaml'))

    # Copy yaml config file and architecture file to the results folder
    results_folder = os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME)
    if rank == 0:
        shutil.copy2(os.path.join('configs', exp_batch, f'{exp_name}.yaml'), results_folder)
        shutil.copy2(os.path.join('network', 'models', 'architectures', 'CIL_multiview', f'{g_conf.MODEL_TYPE}.py'), results_folder)

        # Flush stdout to a log.txt file
    StdoutLogger(os.path.join(results_folder, 'log.txt'), file_mode='a', should_flush=True)

    if len(gpus_list) > 1:
        torch.distributed.barrier()
    
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
    if rank == 0:
        print(torch.cuda.device_count(), 'GPUs to be used: ', gpus_list)

    # Final setup
    set_type_of_process('train_val', root=os.environ["TRAINING_RESULTS_ROOT"], rank=rank, ddp=len(gpus_list) > 1)
    seed_everything(seed=g_conf.MAGICAL_SEED)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True

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

    model = Models(g_conf.MODEL_CONFIGURATION, rank)

    if rank == 0:
        print("===================== Model Configuration =====================")
        num_params = 0
        num_trainable_params = 0
        for param in model.parameters():
            if param.requires_grad:
                num_trainable_params += param.numel()
            num_params += param.numel()
        print(f'Total params in model: {num_params}; trainable: {num_trainable_params} ({100 * num_trainable_params / num_params:.2f}%)')

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=g_conf.LEARNING_RATE)
    if (len(gpus_list) > 1 and g_conf.DATA_PARALLEL):
        if rank == 0:
            print("Using multiple GPUs parallel! ")
        # model = DataParallelWrapper(model)
        # gpus_list_int = [int(el) for el in gpus_list]
        model.to(device_id)
        model = DataParallelDPPWrapper(model, device_ids=[device_id], find_unused_parameters=False)

    # To load a specific checkpoint
    if g_conf.LOAD_CHECKPOINT:
        latest_checkpoint = os.path.join(results_folder, 'checkpoints', g_conf.LOAD_CHECKPOINT)

    # To train model from scratch, or to resume training on a previous one
    elif g_conf.TRAINING_RESUME:
        latest_checkpoint = check_saved_checkpoints(os.path.join(results_folder, 'checkpoints'))
    elif g_conf.FINETUNE:
        latest_checkpoint = None
        finetune_checkpoint = torch.load(g_conf.FINETUNE_MODEL)
        pretrained_dict = finetune_checkpoint['model']

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)

        if rank == 0:
            print('')
            print('    Finetunning model from -> ', g_conf.FINETUNE_MODEL)
    else:
        latest_checkpoint = None


    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint)
        pretrained_dict = checkpoint['model']

        if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        # we manually move optimizer state to GPU memory after loading it from the checkpoint
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if len(gpus_list) > 1:
                        state[k]=v.to(f'cuda:{model.device_ids[0]}')
                    else:
                        state[k]=v.cuda()
        if rank == 0:
            for param_group in optimizer.param_groups:
                print('')
                print('    Resume training from epoch -> ', checkpoint['epoch'])
                print('    Resume the latest learning rate -> ', param_group['lr'])
                if g_conf.LEARNING_RATE_DECAY and g_conf.LEARNING_RATE_SCHEDULE == 'step':
                    print('      - learning rate decay at epoch', g_conf.LEARNING_RATE_DECAY_EPOCHES, ', minimum lr:', g_conf.LEARNING_RATE_MINIMUM)
                print('')
                print('=======================================================================================')
                print('')

        model._current_iteration = checkpoint['iteration'] + 1
        model._done_epoch = checkpoint['epoch']
    else:
        if rank == 0:
            print('')
            print('    Training from scratch')
            print('    Initial learning rate -> ', g_conf.LEARNING_RATE)
            if g_conf.LEARNING_RATE_DECAY and g_conf.LEARNING_RATE_SCHEDULE == 'step':
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
