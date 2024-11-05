# PhysicsAwareCombinatorialASP
This repo contains the implementation of physics-aware assembly sequence planning (ASP) for combinatorial Lego assembly.

## Dependencies
* [Gurobi](https://www.gurobi.com/)
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* [PyTorch](https://pytorch.org/)

## Overview
This work focuses on planning physcially feasible assembly sequence for combinatorial assembly.
Given the 3D shape, the policy outputs a sequence of actions for placing unit primitives to build the goal object. Importantly, the action for each step is physically feasible.
Examples of planned assembly sequences are shown below.

<p >
	<img src="./images/asp.gif" alt="image" width="90%" height="auto" hspace=1>
    <figcaption align="left"></figcaption>
</p>



## Execution
1. Configurate the config file `./config.json`.
    * `Train`: `0`: build a model. `1`: train the policy.
    * `data_folder`: path to the dataset folder.
    * `workspace_dimension`: the XYZ dimension of the building workspace.
    * `output_dir`: If training, this is the directory to save the training results. If building, this is the directory to load the policy model.
    * `trial`: an integer trial number.
    * `build_file_idx`: an index indicating which structure to build in the dataset. No effect when training.
    * `max_step`: task horizon of the ASP. No effect when building.
2. `python3 main.py`.


## Citation
If you find this repository helpful, please kindly cite our work.
```
@article{liu2024physics,
  title={Physics-Aware Combinatorial Assembly Planning using Deep Reinforcement Learning},
  author={Liu, Ruixuan and Chen, Alan and Zhao, Weiye and Liu, Changliu},
  journal={arXiv preprint arXiv:2408.10162},
  year={2024}
}

```

## License
This project is licensed under the MIT License.
