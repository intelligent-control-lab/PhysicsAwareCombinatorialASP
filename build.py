from utils_gym import *
from AssemblyEnvGym import AssemblyEnvironment
from observation_feature_extractor import Policy
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import warnings
warnings.filterwarnings("ignore")

def build(dataset_folder, model_dir, trial_num, file_idx, dimension):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def mask_fn(gymenv: gym.Env) -> np.ndarray:
        return gymenv.valid_action_mask_ui()

    dataset = dataset_folder
    trial = trial_num
    x_size, y_size, z_size = dimension
    max_step = 200

    unique_pid = os.getpid()
    print("Process ID:", unique_pid)
    env = AssemblyEnvironment(root_folder=dataset, X_SIZE=x_size, Y_SIZE=y_size, Z_SIZE=z_size, max_steps=max_step)
    env = ActionMasker(env, action_mask_fn=mask_fn)
    env = Monitor(env)

    mlp_arch = [128]  # Define MLP structure
    policy_kwargs = dict(
        normalize_images=False,
        share_features_extractor=False,
        features_extractor_class=Policy,
        features_extractor_kwargs=dict(features_dim=512),
        activation_fn=torch.nn.Tanh, 
        net_arch=mlp_arch
    )
    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, env, policy_kwargs=policy_kwargs, device=device)
    model.set_parameters(model_dir + "/trial_" + str(trial) + "/asp.zip", device=device)
    
    env.set_fidx(file_idx)
    obs, _  = env.reset()
    print("Building ", env.get_fname())
    cur_state = env.get_full_state()
    done = cur_state["done"]
    truncate = cur_state["truncate"]

    task_graph = dict()
    step = 1
    while(not done and not truncate):
        action_masks = env.valid_action_mask_ui()
        action, state = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, _, done, truncate, info = env.step(action)
        cur_state = env.get_full_state()

        action = env.get_all_actions()[action]
        task_graph[str(step)] = {"x":action[0], "y":action[1], "z":action[2], "brick_id":action[3], "ori":action[4]}
        step += 1
    env.reset()

    build_status = (done and not truncate)
    write_json(task_graph, model_dir + "/trial_" + str(trial) + "/task_graph.json")
    return build_status

        