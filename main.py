# The main file to show how to use the vision algorithms here.
import os
import argparse
import torch

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '--process-type',
        default=None,
        type=str,
        required=True
    )

    argparser.add_argument(
        '--gpus',
        nargs='+',
        dest='gpus',
        type=str,
        required=True
    )

    argparser.add_argument(
        '-f',
        '--folder',
        default='NoDate',
        dest='folder',
        type=str,
        help='The folder of the configuration files'
    )
    argparser.add_argument(
        '-e',
        '--exp',
        default=None,
        dest='exp',
        type=str,
        help='The experiment name of the configuration file'
    )

    args = argparser.parse_args()

    if args.gpus:
        # Check if the vector of GPUs passed are valid.
        for gpu in args.gpus:
            try:
                int(gpu)
            except ValueError:  # Reraise a meaningful error.
                raise ValueError("GPU is not a valid int number")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpus)  # This must to be ahead of the whole excution
    else:
        raise ValueError('You need to define the ids of GPU you want to use by adding: --gpus')


    from console import execute_train_val, execute_val
    if args.process_type is not None:
        if args.process_type == 'train_val':
            if args.exp is None:
                raise ValueError("You should set the exp alias")
            execute_train_val(gpus_list=args.gpus, exp_batch=args.folder, exp_alias=args.exp)

        elif args.process_type == 'val_only':
            if args.exp is None:
                raise ValueError("You should set the exp alias")
            execute_val(gpus_list=args.gpus, exp_batch=args.folder, exp_alias=args.exp)

        else:
            raise Exception("Invalid name for --process-type, chose from (train_val, train_only, val_only)")

    else:
        raise Exception(
            "You need to define the process type with argument '--process-type': train_val, train_only, val_only")

