import os
import torch
from configs import g_conf, set_type_of_process, merge_with_yaml
from network.models_console import Models
from _utils.training_utils import seed_everything, DataParallelWrapper, check_saved_checkpoints_in_total
from _utils.utils import write_model_results, draw_offline_evaluation_results, eval_done

def val_task(model):
    """
    Upstream task is for evaluating your model

    """

    model.eval()
    results_dict = model._eval(model._current_iteration, model._done_epoch)
    if results_dict is not None:
        write_model_results(g_conf.EXP_SAVE_PATH, model.name,
                            results_dict, acc_as_action=g_conf.ACCELERATION_AS_ACTION)
        draw_offline_evaluation_results(g_conf.EXP_SAVE_PATH, metrics_list=g_conf.EVAL_DRAW_OFFLINE_RESULTS_GRAPHS,
                                                    x_range=g_conf.EVAL_SAVE_EPOCHES)

    else:
        raise ValueError('No evaluation results !')


# The main function maybe we could call it with a default name
def execute(gpus_list, exp_batch, exp_name):
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
    set_type_of_process('val_only', root= os.environ["TRAINING_RESULTS_ROOT"])
    seed_everything(seed=g_conf.MAGICAL_SEED)

    model = Models(g_conf.MODEL_TYPE, g_conf.MODEL_CONFIGURATION)
    if len(gpus_list) > 1 and g_conf.DATA_PARALLEL:
        print("Using multiple GPUs parallel! ")
        model = DataParallelWrapper(model)
    # Check if we train model from zero, or we keep training on a previous model
    all_checkpoints = check_saved_checkpoints_in_total(os.path.join(os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,
                                                             g_conf.EXPERIMENT_NAME, 'checkpoints'))
    if all_checkpoints is not None:
        for eval_checkpoint in all_checkpoints:
            if int(eval_checkpoint.split('_')[-2]) in g_conf.EVAL_SAVE_EPOCHES:
                if not eval_done(os.path.join(
                        os.environ["TRAINING_RESULTS_ROOT"], '_results', g_conf.EXPERIMENT_BATCH_NAME,g_conf.EXPERIMENT_NAME),
                                 g_conf.VALID_DATASET_NAME,
                        eval_checkpoint.split('_')[-2]):
                    checkpoint = torch.load(eval_checkpoint)
                    if isinstance(model, torch.nn.DataParallel):
                        model.module.load_state_dict(checkpoint['model'])
                    else:
                        model.load_state_dict(checkpoint['model'])

                    model._current_iteration = checkpoint['iteration'] + 1
                    model._done_epoch = checkpoint['epoch']
                    print('')
                    print('---------------------------------------------------------------------------------------')
                    print('')
                    print('Evaluating epoch:', str(checkpoint['epoch']))

                    model.cuda()
                    val_task(model)
        print('')
        print('---------------------------------------------------------------------------------------')
        print('Evaluation finished !!')
    else:
        print('')
        print('---------------------------------------------------------------------------------------')
        print('No checkpoints to be evaluated. Done')



