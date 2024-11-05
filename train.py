from utils_gym import *
from AssemblyEnvGym import AssemblyEnvironment
from observation_feature_extractor import Policy
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
import warnings
warnings.filterwarnings("ignore")


def train(dataset, dimension, output_path="./logs", num_iters=30000, 
          gamma=0.95, gae_lambda=0.95, ent_coef=0.1, vf_coef=1, lr=0.0003, clip_range=0.5, n_epochs=4, n_steps=256, batch_size=128, n_env=4, max_step=60, trial_num=1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def mask_fn(gymenv: gym.Env) -> np.ndarray:
        return gymenv.valid_action_mask_ui()

    trial = str(trial_num)
    x_size, y_size, z_size = dimension
    gym.envs.registration.register(
            id='optionalSpaceName/lego-asp',
            entry_point=AssemblyEnvironment,
            max_episode_steps=max_step,
            kwargs={'root_folder' : dataset, "max_steps":max_step,
                    "X_SIZE":x_size, "Y_SIZE":y_size, "Z_SIZE":z_size},
        )
    unique_pid = os.getpid()
    print("Process ID:", unique_pid)

    vec_env = make_vec_env('optionalSpaceName/lego-asp', wrapper_class=ActionMasker, wrapper_kwargs=dict(action_mask_fn=mask_fn), n_envs=n_env)

    eval_env = AssemblyEnvironment(root_folder="./"+dataset, X_SIZE=x_size, Y_SIZE=y_size, Z_SIZE=z_size, max_steps=max_step)
    eval_env = ActionMasker(eval_env, action_mask_fn=mask_fn)
    eval_env = Monitor(eval_env)

    evalall_env = AssemblyEnvironment(root_folder="./"+dataset, auto_next_file=1, X_SIZE=x_size, Y_SIZE=y_size, Z_SIZE=z_size, max_steps=max_step)
    evalall_env = ActionMasker(evalall_env, action_mask_fn=mask_fn)
    evalall_env = Monitor(evalall_env)
    check_env(eval_env)
    num_actions = eval_env.get_num_actions()
    mlp_arch = [128]  # Define MLP structure

    policy_kwargs = dict(
            normalize_images=False,
            share_features_extractor=False,
            features_extractor_class=Policy,
            features_extractor_kwargs=dict(features_dim=512),
            activation_fn=torch.nn.Tanh, 
            net_arch=mlp_arch
        )

    print("Num actions: ", num_actions)
    print("Network arch: ", mlp_arch)
    print("n_steps:", n_steps)
    log_path = output_path + "/trial_" + trial + "/"
    logger = configure(log_path, ["log", "json"])

    model = MaskablePPO(MaskableMultiInputActorCriticPolicy, vec_env, 
                        gamma=gamma,
                        verbose=1, 
                        n_steps=n_steps, 
                        gae_lambda=gae_lambda,
                        ent_coef=ent_coef,
                        vf_coef=vf_coef, 
                        learning_rate=lr, 
                        clip_range=clip_range, 
                        n_epochs=n_epochs,
                        policy_kwargs=policy_kwargs, device=device, batch_size=batch_size)  # PPO training
    model.set_logger(logger)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.99999, verbose=1) 

    # Train
    cur_f_idx = 0
    for i in range(50000):
        vec_env.env_method("set_fidx", cur_f_idx)
        eval_env.set_fidx(cur_f_idx)
        eval_env.reset()
        print(i, eval_env.get_fname())
        evalall_env.reset()
        
        evalall_callback = MaskableEvalCallback(evalall_env, 
                                                eval_freq=n_steps, 
                                                n_eval_episodes=len(eval_env.get_all_fnames()), 
                                                verbose=0)
        eval_callback = MaskableEvalCallback(eval_env, 
                                             eval_freq=n_steps, 
                                             n_eval_episodes=1, 
                                             callback_on_new_best=callback_on_best,
                                             verbose=0)
        event_callback = CallbackList([eval_callback, evalall_callback])
        model.learn(total_timesteps=num_iters, progress_bar=True, callback=event_callback, reset_num_timesteps=False)
        model.save(log_path+"asp.zip")

        cur_f_idx += 1
        if(cur_f_idx >= len(eval_env.get_all_fnames())):
            cur_f_idx = 0
