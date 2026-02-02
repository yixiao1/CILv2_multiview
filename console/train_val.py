import os
import torch
import torch.distributed as dist
import time
import shutil
import numpy as np
from configs import g_conf, set_type_of_process, merge_with_yaml
from network.models_console import Models
from _utils.training_utils import seed_everything, DataParallelWrapper, DataParallelDPPWrapper, check_saved_checkpoints, update_learning_rate
from _utils.utils import extract_targets, extract_other_inputs, extract_commands, print_train_info, test_stop, add_wp_to_image, build_pose_matrix
from _utils.evaluation import evaluation_saving
from logger import _logger
from dataloaders.transforms import inverse_normalize
import torch.profiler as profiler


def update_early_stopping(flags, rank, world_size):
    torch.cuda.set_device(rank)
    data_dpp = {
        'flags': flags,
        'rank': rank
    }
    outputs_dpp = [None for _ in range(world_size)]
    dist.all_gather_object(outputs_dpp, data_dpp)    # we only want to operate on the collected objects at master node

    flags = [el['flags'] for el in outputs_dpp if el['rank'] == 0][0]


def train_upstream_task(model, optimizer, rank=0, world_size=1, stop_iter=-1, prof_obj=None):
    """
    Upstream task is for training your model

    """
    early_stopping_flags = []
    acc_time = 0.0
    time_start = time.time()
    local_iteration = 0
    init_iteration = model._current_iteration
    init_epoch = (model._current_iteration * g_conf.BATCH_SIZE // len(model))
    steering_hook = None

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
                # src_images = [[data['current'][i][camera_type] for camera_type in g_conf.DATA_USED] for i in range(len(data['current']))]
                src_images = torch.stack([torch.stack([data['current'][i][camera_type] for camera_type in g_conf.DATA_USED], dim=1) for i in range(len(data['current']))], dim=1)  # [B, S, cam, 3, H, W]
                src_images = src_images.cuda(non_blocking=True).to(f'cuda:{model.device_ids[0]}')
                src_directions = [extract_commands(data['current'][i]['can_bus']['direction']).to(f'cuda:{model.device_ids[0]}') for i in
                                  range(len(data['current']))]
                src_s = [extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                         ignore=['direction']).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['future']))]
                else:
                    tgt_a = [extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).to(f'cuda:{model.device_ids[0]}') for i in range(len(data['current']))]
            else:
                src_images = torch.stack([torch.stack([data['current'][i][camera_type] for camera_type in g_conf.DATA_USED], dim=1) for i in range(len(data['current']))], dim=1)  # [B, S, cam, 3, H, W]
                src_images = src_images.cuda(non_blocking=True)
                src_directions = [extract_commands(data['current'][i]['can_bus']['direction']).cuda(non_blocking=True) for i in
                                  range(len(data['current']))]
                src_s = [extract_other_inputs(data['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                         ignore=['direction']).cuda(non_blocking=True) for i in range(len(data['current']))]
                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [extract_targets(data['future'][i]['can_bus_future'], g_conf.TARGETS).cuda(non_blocking=True) for i in range(len(data['future']))]
                else:
                    tgt_a = [extract_targets(data['current'][i]['can_bus'], g_conf.TARGETS).cuda(non_blocking=True) for i in range(len(data['current']))]
            tgt_a[0][:, 0] = tgt_a[0][:, 0] * 2.


            # src_images = src_images.to(f'cuda:{model.device_ids[0]}')
            # src_directions = src_directions.to(f'cuda:{model.device_ids[0]}')
            # src_s = src_s.to(f'cuda:{model.device_ids[0]}')
            # model.to(f'cuda:{model.device_ids[0]}')
            if g_conf.MODEL_TYPE == 'CILv2_multiview_TD_Diffusion_attention':
                inp_tgt = tgt_a[0].unsqueeze(1)
                outputs_diffusion = model.forward(src_images, src_directions, src_s, targets=inp_tgt)
                action_outputs = outputs_diffusion["denoise_pred"]
                loss = outputs_diffusion["action_loss"]
                acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss, loss, loss)
            else:
                action_outputs = model.forward(src_images, src_directions, src_s)
                loss_params = {
                    'action_output': action_outputs,
                    'targets_action': tgt_a,
                    'variable_weights': g_conf.LOSS_WEIGHT,
                    'input_speed': src_s
                }

                if g_conf.ACCELERATION_AS_ACTION:
                    if g_conf.LOSS == 'Action2WP_L1':
                        loss, steer_loss, acceleration_loss, wp_loss = model.loss(loss_params)
                    else:
                        loss, steer_loss, acceleration_loss = model.loss(loss_params)
                    if rank == 0:
                        if g_conf.LOSS == 'Action2WP_L1':
                            acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss, steer_loss, acceleration_loss, wp_loss_data=wp_loss)
                        else:
                            acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss, steer_loss, acceleration_loss)
                else:
                    loss, steer_loss, throttle_loss, brake_loss = model.loss(loss_params)
                    if rank == 0:
                        acc_time = print_train_info(g_conf.TRAIN_PRINT_LOG_FREQUENCY, g_conf.NUMBER_EPOCH, g_conf.BATCH_SIZE, model, time_start,
                                                    acc_time, loss, steer_loss, throttle_loss, brake_loss)


            time_start = time.time()

            optimizer.zero_grad()

            if g_conf.SPEED_AUGMENTATION:
                mask_steer = (tgt_a[0][:, 0] != -1000.0).detach()
                steering_hook = model._model.register_steering_mask(mask_steer)

            loss.backward()
            optimizer.step()

            if g_conf.SPEED_AUGMENTATION and steering_hook is not None:
                steering_hook.remove()



            """
            ################################################
                Adding tensorboard logs
            #################################################
            """
            if rank == 0 and model._current_iteration % 100 == 0 :
                _logger.add_scalar('Loss', loss.item(), model._current_iteration)

                ## Adding loss to tensorboard
                if g_conf.MODEL_TYPE == 'CILv2_multiview_TD_Diffusion_attention':
                    _logger.add_scalar('Loss_diffusion', loss.item(), model._current_iteration)
                else:
                    if g_conf.LOSS == 'Action2WP_L1':
                        _logger.add_scalar('Loss_wp', wp_loss.item(), model._current_iteration)
                        _logger.add_scalar('Weighted Loss_wp', g_conf.LOSS_WEIGHT['actions']['wp'] * wp_loss.item(), model._current_iteration)
                    _logger.add_scalar('Loss_steer', steer_loss.item(), model._current_iteration)
                    _logger.add_scalar('Weighted Loss_steer', g_conf.LOSS_WEIGHT['actions']['steer'] * steer_loss.item(), model._current_iteration)
                    if g_conf.ACCELERATION_AS_ACTION:
                        _logger.add_scalar('Loss_acceleration', acceleration_loss.item(), model._current_iteration)
                        _logger.add_scalar('Weighted Loss_acceleration', g_conf.LOSS_WEIGHT['actions']['acceleration'] * acceleration_loss.item(), model._current_iteration)
                    else:
                        _logger.add_scalar('Loss_throttle', throttle_loss.item(), model._current_iteration)
                        _logger.add_scalar('Loss_brake', brake_loss.item(), model._current_iteration)

                if (g_conf.ADD_WP_PREDICTIOS_LOG and model._current_iteration % 1000 == 0):
                    # src_iamges0 = [255. * np.swapaxes(np.swapaxes(img[0, :, :, :].cpu().numpy(), 0, 2), 0, 1) for img in src_images[0]]
                    # src_iamges0 = [g_conf.IMG_NORMALIZATION['mean'] + g_conf.IMG_NORMALIZATION['std'] * np.swapaxes(np.swapaxes(img[0, :, :, :].cpu().numpy(), 0, 2), 0, 1) for img in src_images[0]]

                    src_images0 = [inverse_normalize(img, g_conf.IMG_NORMALIZATION['mean'], g_conf.IMG_NORMALIZATION['std']) for img in src_images[0]]
                    src_iamges0 = [255. * np.swapaxes(np.swapaxes(img[0, :, :, :].cpu().numpy(), 0, 2), 0, 1) for img in src_images0]
                    cam_K = [np.array([[g_conf.CAM_FOCAL[indx][0], 0, g_conf.CAM_CENTER_POINT[indx][0], 0], [0, g_conf.CAM_FOCAL[indx][1], g_conf.CAM_CENTER_POINT[indx][1], 0], [0, 0, 1, 0]]) for indx in range(len(g_conf.CAM_FOCAL))]
                    cam_T = [build_pose_matrix(g_conf.CAM_ROTATION[indx], g_conf.CAM_TRANSLATION[indx]) for indx in range(len(g_conf.CAM_ROTATION))]
                    speed_denorm = src_s[0][0].cpu().numpy()[0] * (g_conf.DATA_NORMALIZATION['speed'][1] - g_conf.DATA_NORMALIZATION['speed'][0]) + g_conf.DATA_NORMALIZATION['speed'][0]
                    accel_denorm_gt = tgt_a[0][0, 1].cpu().numpy() * 6.0
                    steer_denorm_gt = tgt_a[0][0, 0].cpu().numpy() * 3.14159265359 if tgt_a[0][0, 0].cpu().numpy() != -1000.0 else 0.0
                    accel_denorm = action_outputs[0,: , 1].detach().cpu().numpy()[0] * 6.0
                    steer_denorm = action_outputs[0, :, 0].detach().cpu().numpy()[0] * 3.14159265359 / 2.

                    # wp_image = add_wp_to_image(src_iamges0, action_outputs[0, :, 0].detach().cpu().numpy(), action_outputs[0,: , 1].detach().cpu().numpy(), src_s[0][0].cpu().numpy(), cam_T, cam_K, cam_size=g_conf.CAM_IM_SIZE, xi=g_conf.CAM_XI)
                    wp_image_gt = add_wp_to_image(src_iamges0, steering_rad=steer_denorm_gt, acceleration=accel_denorm_gt, speed=speed_denorm, cam_T=cam_T, cam_K=cam_K, cam_size=g_conf.CAM_IM_SIZE, xi=g_conf.CAM_XI)
                    wp_image_predictions = add_wp_to_image(src_iamges0, steering_rad=steer_denorm, acceleration=accel_denorm, speed=speed_denorm, cam_T=cam_T, cam_K=cam_K, cam_size=g_conf.CAM_IM_SIZE, xi=g_conf.CAM_XI)
                    _logger.add_image('WP GT', wp_image_gt, model._current_iteration)
                    _logger.add_image('WP Predictions', wp_image_predictions, model._current_iteration)
                    _logger.add_scalar('Steering GT', steer_denorm_gt, model._current_iteration)
                    _logger.add_scalar('Acceleration GT', accel_denorm_gt, model._current_iteration)
                    _logger.add_scalar('Speed', speed_denorm, model._current_iteration)
                    _logger.add_scalar('Steering Predictions', steer_denorm, model._current_iteration)
                    _logger.add_scalar('Acceleration Predictions', accel_denorm, model._current_iteration)

            if prof_obj is not None and rank==0:
                prof_obj.step()

            if test_stop(g_conf.NUMBER_EPOCH * len(model), model._current_iteration * g_conf.BATCH_SIZE) or (model._current_iteration > stop_iter and stop_iter > 0):
                print('')
                print('Training finished !!')
                break

            local_iteration += 1

            model._current_iteration = init_iteration + local_iteration
            model._done_epoch = (model._current_iteration * g_conf.BATCH_SIZE // len(model))

            if world_size > 1:
                dataloader.sampler.set_epoch(model._done_epoch - 1)

            # del src_images
            # del src_directions
            # del tgt_a
            # del src_s
            # del action_outputs
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
        model = DataParallelDPPWrapper(model, device_ids=[device_id], find_unused_parameters=False)

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

        '''
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(pretrained_dict)
        else:
            model.load_state_dict(pretrained_dict)
        '''

        # yi model
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            name = 'module.' + k # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

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

    profile_code = False
    if profile_code:
        prof = None
        warmup_prof = 20
        active_prof = 50
        skip_prof = 20
        stop_iter_prof = warmup_prof + active_prof + skip_prof
        if rank == 0:
            prof = profiler.profile(
                activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                schedule=profiler.schedule(
                    wait=skip_prof,   # skip first 20 iterations (no profiling)
                    warmup=warmup_prof,  # collect but discard (helps stabilize)
                    active=active_prof  # profile next 10 iterations
                    )
                )
            prof.start()
        train_upstream_task(model, optimizer, rank=rank, world_size=len(gpus_list), stop_iter=stop_iter_prof, prof_obj=prof)
        if rank == 0 and prof is not None:
            prof.stop()
            print(prof.key_averages().table(sort_by="cuda_time_total"))

    else:
        train_upstream_task(model, optimizer, rank=rank, world_size=len(gpus_list))

