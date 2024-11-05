import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import glob
import os
import time
import sys
import json
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
import torch
import torch.nn as nn


def load_json(fname):
	f = open(fname)
	content = json.load(f)
	f.close()
	return content

def load_data_fname_from_folder(root_folder):
    data_sets = []

    def load_from_path(path):
        save_export_path = os.path.join(path, 'vox.npy')

        if os.path.exists(save_export_path):
            data_sets.append(save_export_path)
        else:
            for subdir in os.listdir(path):
                subdir_path = os.path.join(path, subdir)
                if os.path.isdir(subdir_path):
                    load_from_path(subdir_path)

    load_from_path(root_folder)
    return data_sets

def write_json(graph, output_dir):
	json_graph = json.dumps(graph, indent=4)
    
    # Writing json
	with open(output_dir, "w") as outfile:
		outfile.write(json_graph)
	outfile.close()
