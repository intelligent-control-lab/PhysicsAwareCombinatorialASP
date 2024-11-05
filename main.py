from train import *
from build import *

if __name__ == '__main__':
    config = load_json("./config.json")
    is_train = config["Train"]
    dataset = config["data_folder"]
    dimension = config["workspace_dimension"]
    output_path = config["output_dir"]
    max_step = config["max_step"]
    trial_num = config["trial"]
    if(is_train):
        num_iters = config["ppo_params"]["num_iters"]
        gamma = config["ppo_params"]["gamma"]
        gae_lambda = config["ppo_params"]["gae_lambda"]
        ent_coef = config["ppo_params"]["ent_coef"]
        vf_coef = config["ppo_params"]["vf_coef"]
        lr = config["ppo_params"]["learning_rate"]
        clip_range = config["ppo_params"]["clip_range"]
        n_epochs = config["ppo_params"]["n_epochs"]
        n_steps = config["ppo_params"]["n_steps"]
        batch_size = config["ppo_params"]["batch_size"]
        n_env = config["ppo_params"]["n_env"]

        train(dataset, dimension, output_path, num_iters, gamma, gae_lambda, ent_coef, vf_coef, lr, 
              clip_range, n_epochs, n_steps, batch_size, n_env, max_step, trial_num)

    else:
        build_file_idx = config["build_file_idx"]
        status = build(dataset, output_path, trial_num, build_file_idx, dimension)
