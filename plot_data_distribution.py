# Get all of the data in a specified directory (and subdirectories) and plot the distribution of it.
# We are interested in the steering, acceleration (plus throttle and brake), speed, and command (direction) to follow.
# The data is stored in the following format in a .json file:
# {"acceleration": -1.0, "brake": 1.0, "direction": 2.0, "hand_brake": false, "reverse": false,
# "speed": 2.1670665238630355e-08, "steer": 0.08580279350280762, "throttle": 0.0}

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from glob import glob
import click
from multiprocessing import Pool
import pandas as pd


# Aux functions
def get_data(data_file):
    """ Get the data from a single data file """
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


@click.command()
@click.option('--dataset-dir', default='.', help='The directory of the dataset')
@click.option('--save-dir', default='plots', help='The directory to save the generated plots')
# Optional arguments
@click.option('--processes-per-cpu', default=2, help='The number of processes per CPU to use for multiprocessing')
def plot_data_distribution(dataset_dir, save_dir, processes_per_cpu):
    """ Plot the distribution of the data with seaborn. The file names are in the format cmd_fix_can_bus{%06d}.json """
    # Set the full path of the dataset directory
    dataset_dir = os.path.abspath(dataset_dir)

    # Create the save directory if it doesn't exist
    save_dir = os.path.join(dataset_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get the list of all the data files in all the dataset_dir directories and subdirectories
    data_files = glob(os.path.join(dataset_dir, '**', 'cmd_fix_can_bus*.json'), recursive=True)

    # Extract the data from the data files with multiprocessing; each process will extract the data from a single file
    with Pool(os.cpu_count() * processes_per_cpu) as pool:
        data = list(tqdm(pool.imap(get_data, data_files), total=len(data_files)))

    # Save the data as a .npy file for easy access
    np.save(os.path.join(save_dir, 'data.npy'), data)

    # Get the dataframes of the data
    df = pd.DataFrame(data)

    # Set a column indicating the command with a string
    command_dict = {1.0: 'Turn Left', 2.0: 'Turn Right', 3.0: 'Go Straight', 4.0: 'Follow Lane'}
    df['command'] = df['direction'].map(command_dict)
    print(df.head())

    # Now, a 2d distribution plot:
    sns.jointplot(data=df, x='steer', y='acceleration', kind='kde', hue='command')
    # Add the whole dataset as a scatter plot
    sns.scatterplot(data=df, x='steer', y='acceleration', hue='command', alpha=0.1)
    plt.savefig(os.path.join(save_dir, 'steer_acceleration_jointplot_bycommand.png'))