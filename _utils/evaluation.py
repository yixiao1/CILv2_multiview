import datetime
from collections import OrderedDict
import time
import torch
import os
from contextlib import contextmanager

from _utils.utils import extract_targets, extract_commands, \
    write_model_results, is_result_better, extract_other_inputs, draw_offline_evaluation_results, eval_done
from configs import g_conf
from logger import _logger
from dataloaders.transforms import inverse_normalize


def get_model_state(model: torch.nn.Module) -> OrderedDict:
    """ Get the state of a model """
    if isinstance(model, torch.nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def save_model(model: torch.nn.Module, optimizer, best_pred: int) -> None:
    """ Save the model to the checkpoint directory """

    model_state_dict = get_model_state(model)
    saving_dict = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'iteration': model._current_iteration - 1,
        'epoch': model._done_epoch,
        'best_pred': best_pred
    }
    torch.save(saving_dict,
               os.path.join(g_conf.EXP_SAVE_PATH, 'checkpoints',
                            f'{model.name}_{model._done_epoch}_{model._current_iteration - 1}.pth'))


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def evaluation_on_model(model, data_loaders, model_name, evaluator, eval_iteration, eval_epoch):
    results_dict = {}
    for data_loader in data_loaders:
        total = len(data_loader.dataset)  # inference data loader must have a fixed length
        dataset_name = data_loader.dataset.dataset_name
        info = f"Start evaluation for model {model_name} on {dataset_name} with {total} images at iteration {eval_iteration} / epoch {eval_epoch}"
        print(info)
        evaluator.reset()
        logging_interval = 500
        start_time = time.time()

        with inference_context(model), torch.no_grad():
            for idx, x in enumerate(data_loader):
                # Extract the inputs
                src_images = [[x['current'][i][camera_type].cuda() for camera_type in g_conf.DATA_USED] for i in range(len(x['current']))]
                src_directions = [extract_commands(x['current'][i]['can_bus']['direction']).cuda() for i in range(len(x['current']))]
                src_speeds = [extract_other_inputs(x['current'][i]['can_bus'], g_conf.OTHER_INPUTS,
                                                   ignore=['direction']).cuda() for i in range(len(x['current']))]

                if g_conf.ENCODER_OUTPUT_STEP_DELAY > 0 or g_conf.DECODER_OUTPUT_FRAMES_NUM != g_conf.ENCODER_INPUT_FRAMES_NUM:
                    tgt_a = [extract_targets(x['future'][i]['can_bus_future'],
                                             g_conf.TARGETS).cuda() for i in range(len(x['future']))]
                else:
                    tgt_a = [extract_targets(x['current'][i]['can_bus'],
                                             g_conf.TARGETS).cuda() for i in range(len(x['current']))]

                action_outputs = model.forward_eval(src_images, src_directions, src_speeds)
                torch.cuda.synchronize()
                evaluator.process(action_outputs, tgt_a)

                """
                ################################################
                    Adding visualization to tensorboard
                #################################################
                """

                if idx in list(range(0, min(g_conf.EVAL_IMAGE_WRITING_NUMBER, len(data_loader)))):
                    # saving only one per batch to save time
                    eval_images = [[x['current'][i][camera_type][:1].cuda() for camera_type in g_conf.DATA_USED] for i
                                   in range(len(x['current']))]
                    eval_directions = [extract_commands(x['current'][i]['can_bus']['direction'])[:1].cuda() for i in
                                       range(len(x['current']))]
                    eval_speeds = [
                        extract_other_inputs(x['current'][i]['can_bus'],
                                             g_conf.OTHER_INPUTS,
                                             ignore=['direction'])[:1].cuda() for i in range(len(x['current']))
                    ]
                    input_frames = []
                    for frame in eval_images:
                        cams = []
                        for i in range(len(g_conf.DATA_USED)):
                            cams.append(inverse_normalize(frame[i], g_conf.IMG_NORMALIZATION['mean'],
                                                          g_conf.IMG_NORMALIZATION['std']).detach().cpu().numpy().squeeze())
                        input_frames.append(cams)

                    # we save only the first of the batch
                    if g_conf.EVAL_SAVE_LAST_Conv_ACTIVATIONS:
                        _logger.add_gradCAM_attentions_to_disk('Valid', model,
                                                               [eval_images, eval_directions, eval_speeds],
                                                               input_rgb_frames=input_frames,
                                                               epoch=eval_epoch,
                                                               save_path=os.path.join(g_conf.EXP_SAVE_PATH, 'Eval',
                                                                                      f'Valid_gradCAM_{dataset_name}'),
                                                               batch_id=idx)

                if (idx + 1) % logging_interval == 0:
                    duration = time.time() - start_time
                    seconds_per_img = duration / ((idx + 1)*g_conf.EVAL_BATCH_SIZE)
                    eta = datetime.timedelta(seconds=int(seconds_per_img * total - duration))
                    info = f"Evaluation done {(idx + 1)*g_conf.EVAL_BATCH_SIZE}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}"
                    print(info)

                del src_images
                del src_directions
                del tgt_a
                del action_outputs

        results = evaluator.evaluate(eval_epoch, dataset_name)
        if results is None:
            results = {}
        results_dict[dataset_name] = results
    return results_dict


def save_model_if_better(results_dict, model, optimizer, save_all=False):
    # we are saving the model if it is better than the previous one
    if g_conf.PROCESS_NAME == 'train_val':
        dataset_name = list(results_dict.keys())[0]
        results = results_dict[dataset_name]
        is_better_flag = is_result_better(g_conf.EXP_SAVE_PATH, model.name, dataset_name)
        # compulsively saving all checkpoints
        if save_all:
            print(f'Checkpoint at iteration {model._current_iteration-1} / epoch {model._done_epoch} is saved')
            if is_better_flag:
                best_pred = results[model.name]['MAE']
                save_model_if_better.best_pred = best_pred
            else:
                best_pred = save_model_if_better.best_pred if hasattr(save_model_if_better, "best_pred") else 0

            # Save the model
            save_model(model, optimizer, best_pred)

        else:
            if is_better_flag:
                print(f'Checkpoint at iteration {model._current_iteration-1} / epoch {model._done_epoch} is saved')
                best_pred = results[model.name]['MAE']
                save_model_if_better.best_pred = best_pred

                # Save the model
                save_model(model, optimizer, best_pred)
            else:
                best_pred = save_model_if_better.best_pred if hasattr(save_model_if_better, "best_pred") else 0

    else:
        raise NotImplementedError('Not found process name !')

    return is_better_flag, best_pred


def evaluation_saving(model, optimizers, early_stopping_flags, save_all_checkpoints = False):
    """
    Evaluates but also saves if the model is better
    """
    if g_conf.PROCESS_NAME == 'train_val':
        if (model._done_epoch in g_conf.EVAL_SAVE_EPOCHES) and \
                ((model._current_iteration-1)*g_conf.BATCH_SIZE <= len(model) * model._done_epoch):

            # check if the checkpoint has been evaluated
            if not eval_done(os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results',
                                          g_conf.EXPERIMENT_BATCH_NAME, g_conf.EXPERIMENT_NAME),
                             g_conf.VALID_DATASET_NAME, model._done_epoch):
                print('\n---------------------------------------------------------------------------------------\n')
                print(f'Evaluating epoch: {model._done_epoch}')
                # switch to evaluation mode
                model.eval()
                results_dict = model._eval(model._current_iteration, model._done_epoch)
                if results_dict is not None:
                    write_model_results(g_conf.EXP_SAVE_PATH, model.name,
                                        results_dict, acc_as_action=g_conf.ACCELERATION_AS_ACTION)
                    draw_offline_evaluation_results(g_conf.EXP_SAVE_PATH,
                                                    metrics_list=g_conf.EVAL_DRAW_OFFLINE_RESULTS_GRAPHS,
                                                    x_range=g_conf.EVAL_SAVE_EPOCHES)
                    is_better_flag, best_pred = save_model_if_better(results_dict, model, optimizers, save_all=save_all_checkpoints)
                    if g_conf.EARLY_STOPPING:
                        early_stopping_flags.append(not is_better_flag)
                    else:
                        early_stopping_flags.append(False)
                else:
                    raise ValueError('No evaluation results !')
                # switch back to train mode and continue training
                model.train()
                print('\n---------------------------------------------------------------------------------------\n')

    else:
        raise NotImplementedError('No this type of process name defined')

    return early_stopping_flags
