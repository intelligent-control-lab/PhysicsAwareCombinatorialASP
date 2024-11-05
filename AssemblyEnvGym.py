from utils_gym import *
sys.path.insert(1, './StableLego/py_scripts')
from stability_analysis_graph_input import *

class AssemblyEnvironment(gym.Env):
    def __init__(self, root_folder="./dataset", auto_next_file=0,
                 max_steps=60, X_SIZE=48, Y_SIZE=48, Z_SIZE=48, AVAILABLE_BRICK_IDS=[2, 3, 4, 5, 6, 9, 10, 12]):
        super().__init__()

        print(root_folder)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.lego_lib = load_json("./lego_library.json")

        self.X_SIZE, self.Y_SIZE, self.Z_SIZE = X_SIZE, Y_SIZE, Z_SIZE
        self.AVAILABLE_BRICK_IDS = AVAILABLE_BRICK_IDS
        self.AVAILABLE_ORIENTATIONS = [0, 1]
        self.dataset_fnames = load_data_fname_from_folder(root_folder)
        print("Num data:", len(self.get_all_fnames()))
        self.pick_idx = np.random.randint(0, len(self.dataset_fnames))

        self.max_steps = max_steps
        self.auto_next_file = auto_next_file
        self.actions = self.generate_all_actions()
        self.num_actions = self.get_num_actions()
        self.big_reward = 0
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict({
            "voxel_state": spaces.Box(low=0, high=1, shape=(2, self.X_SIZE, self.Y_SIZE, self.Z_SIZE), dtype=np.uint8),
            "inventory_state":spaces.Box(low=0, high=1, shape=(max(self.AVAILABLE_BRICK_IDS)+1,), dtype=np.uint8)
        })
        self.initial_mask = dict()
        self.reset()
        self.fname = ""
    
    def set_fidx(self, idx):
        self.pick_idx = idx

    def reset(self, seed=None, options=None):
        if(self.auto_next_file):
            self.pick_idx += 1
            if(self.pick_idx >= len(self.dataset_fnames)):
               self.pick_idx = 0
        pick_idx = self.pick_idx
        data_fname = self.dataset_fnames[pick_idx]
        self.fname = (pick_idx, data_fname)
        pre_action_idx = -1
        load_voxel = np.load(data_fname)
        assert(load_voxel.shape[0] <= self.X_SIZE)
        assert(load_voxel.shape[1] <= self.Y_SIZE)
        assert(load_voxel.shape[2] <= self.Z_SIZE)
        target_voxel = np.zeros((self.X_SIZE, self.Y_SIZE, self.Z_SIZE), dtype=np.uint8)
        target_voxel[:load_voxel.shape[0], :load_voxel.shape[1], :load_voxel.shape[2]] = load_voxel[:, :, :]
        current_voxel = np.zeros((self.X_SIZE, self.Y_SIZE, self.Z_SIZE), dtype=np.uint8)
        
        self.possible_actions = self.generate_all_possible_actions(target_voxel)
        cur_step = 0
        assembled_bricks = dict()
        voxel_state = np.zeros((2, self.X_SIZE, self.Y_SIZE, self.Z_SIZE), dtype=np.uint8)
        voxel_state[0, :, :, :] = current_voxel
        voxel_state[1, :, :, :] = target_voxel[:self.X_SIZE, :self.Y_SIZE, :self.Z_SIZE]
        inventory_state = np.zeros(max(self.AVAILABLE_BRICK_IDS)+1, dtype=np.uint16)
        for brick_id in self.AVAILABLE_BRICK_IDS:
            inventory_state[brick_id] = self.lego_lib[str(brick_id)]['inventory']

        self.full_state = {"voxel_state":voxel_state,
                           "inventory_state":inventory_state,
                           "cur_step":cur_step,
                           "assembled_task_graph":assembled_bricks,
                           "reward":0,
                           "pre_action":pre_action_idx,
                           "done":False,
                           "truncate":False,
                           "stability":0}
        
        inventory_state_uint8 = np.copy(inventory_state)
        inventory_state_uint8 = np.clip(inventory_state_uint8, 0, 1)
        self.state = {"voxel_state":np.copy(voxel_state),
                      "inventory_state":np.asarray(inventory_state_uint8, dtype=np.uint8),
                      }
        return self.state, {}

    def get_env_device(self):
        return self.device
    
    def get_max_steps(self):
        return self.max_steps
    
    def get_all_fnames(self):
        return self.dataset_fnames
    
    def valid_action_mask_ui(self):
        mask = self.valid_action_mask(self.full_state)
        self.mask = mask
        return mask
    
    def valid_action_mask(self, state):
        valid_actions = np.zeros(len(self.actions), dtype=bool) # Start by assuming all actions are invalid
        cur_voxel = state["voxel_state"][0, :, :, :]
        target_voxel = state["voxel_state"][1, :, :, :]
        inventory = state["inventory_state"]
       
        for i in range(len(self.possible_actions)):
            cur_x, cur_y, cur_z, brick_id, ori, action_idx = self.possible_actions[i]
            width, height = self.lego_lib[str(brick_id)]["width"], self.lego_lib[str(brick_id)]["height"]
            if(ori == 1):
                width, height = height, width
            valid_actions[action_idx] = True

            # Check if the brick would overlap with existing bricks
            if np.any(cur_voxel[cur_x:cur_x + height, cur_y:cur_y + width, cur_z]):
                valid_actions[action_idx] = False
            # No connections
            elif(cur_z > 0 and 
                 np.all(cur_voxel[cur_x:cur_x + height, cur_y:cur_y + width, cur_z+1] == 0) and 
                 np.all(cur_voxel[cur_x:cur_x + height, cur_y:cur_y + width, cur_z-1] == 0)):
                valid_actions[action_idx] = False
            # Check inventory for the brick
            elif inventory[brick_id] <= 0:
                valid_actions[action_idx] = False
            # Cannot place when both top and bottom are occupied
            elif(self.top_layer_occupied(cur_x, cur_y, cur_z, height, width, cur_voxel) and 
                 self.bottom_layer_occupied(cur_x, cur_y, cur_z, height, width, cur_voxel)):
                valid_actions[action_idx] = False
            # Cannot block upper and lower bricks
            elif(self.block_future_bricks(cur_x, cur_y, cur_z, height, width, cur_voxel, target_voxel)):
                valid_actions[action_idx] = False
            # Structural stability
            if(valid_actions[action_idx]):
                next_state = self.simulate(state, action_idx)
                if(cur_z == 0):
                    valid_actions[action_idx] = True
                elif(not self.structure_stable(next_state["assembled_task_graph"])):
                    valid_actions[action_idx] = False
        return valid_actions

    def top_layer_occupied(self, x, y, z, h, w, cur_voxel):
        occupied = False
        if(z != self.Z_SIZE - 1 and np.any(cur_voxel[x:min(x + h, self.X_SIZE), y:min(y + w, self.Y_SIZE), z+1])):
            occupied = True
        return occupied
    
    def bottom_layer_occupied(self, x, y, z, h, w, cur_voxel):
        occupied = False
        if(z == 0 or np.any(cur_voxel[x:min(x + h, self.X_SIZE), y:min(y + w, self.Y_SIZE), z-1])):
            occupied = True
        return occupied

    def block_future_bricks(self, cur_x, cur_y, cur_z, height, width, cur_voxel, target_voxel):
        valid_mask = np.zeros((height, width))
        for i in range(cur_x, cur_x + height):
            for j in range(cur_y, cur_y + width):
                unoccupied_lower_z = -1
                for k in range(cur_z, -1, -1):
                    if(target_voxel[i, j, k] == 0):
                        unoccupied_lower_z = k
                        break
                unoccupied_upper_z = 0
                for k in range(cur_z, self.Z_SIZE):
                    if(target_voxel[i, j, k] == 0):
                        unoccupied_upper_z = k
                        break
                x = i - cur_x
                y = j - cur_y
                valid_mask[x, y] = ((cur_z == 0 or cur_voxel[i, j, cur_z-1] == 1 or (unoccupied_lower_z >= 0 and np.all(cur_voxel[i, j, unoccupied_lower_z+1:cur_z] == 0))) and
                                    (np.all(cur_voxel[i, j, cur_z+1:unoccupied_upper_z] == 0) or cur_voxel[i, j, cur_z+1]))                           
        return np.sum(valid_mask) != height * width
    
    def generate_all_actions(self):
        all_actions = []
        for brick_id in self.AVAILABLE_BRICK_IDS:
            width, height = self.lego_lib[str(brick_id)]["width"], self.lego_lib[str(brick_id)]["height"]
            for ori in self.AVAILABLE_ORIENTATIONS:
                if(width == height and ori == self.AVAILABLE_ORIENTATIONS[1]):
                    continue
                for k in range(self.Z_SIZE):
                    for j in range(self.Y_SIZE):
                        for i in range(self.X_SIZE):
                            cur_x, cur_y, cur_z = i, j, k
                            valid = True
                            # Check if placing the brick would go out of bounds
                            if(cur_x + height > self.X_SIZE or cur_y + width > self.Y_SIZE or cur_z >= self.Z_SIZE):
                                valid = False
                            # Only store valid actions
                            if(valid):
                                all_actions.append((i, j, k, brick_id, ori))
        return all_actions
    
    def generate_all_possible_actions(self, target_voxel):
        pick_idx = self.fname[0]
        self.mask = np.ones(len(self.actions), dtype=bool)
        if(pick_idx in self.initial_mask.keys()):
            return self.initial_mask[pick_idx]
        all_actions = []
        for i in range(len(self.actions)):
            cur_x, cur_y, cur_z, brick_id, ori = self.actions[i]
            width, height = self.lego_lib[str(brick_id)]["width"], self.lego_lib[str(brick_id)]["height"]
            if(ori == 1):
                width, height = height, width
            if(np.sum(target_voxel[cur_x:cur_x+height, cur_y:cur_y+width, cur_z]) != height * width):
                continue
            else:                
                all_actions.append((cur_x, cur_y, cur_z, brick_id, ori, i))
        self.initial_mask[pick_idx] = all_actions
        return all_actions

    def get_fname(self):
        return self.fname

    def get_num_actions(self):
        return len(self.actions)
    
    def get_all_actions(self):
        return self.actions

    def get_state(self):
        return self.state
    
    def get_full_state(self):
        return self.full_state
    
    def simulate(self, cur_state, new_action):
        pre_action_idx = cur_state["pre_action"]
        prev_action = self.actions[pre_action_idx]
        cur_action = self.actions[new_action]
        cur_voxel = np.copy(cur_state["voxel_state"][0, :, :, :])
        target_voxel = cur_state["voxel_state"][1, :, :, :]
        assembled_task_graph = copy.deepcopy(cur_state["assembled_task_graph"])
        inventory = np.copy(cur_state["inventory_state"])
        current_step = cur_state["cur_step"]
        
        new_x, new_y, new_z, place_brick_id, place_brick_ori = cur_action
        
        width, height = self.lego_lib[str(place_brick_id)]['width'], self.lego_lib[str(place_brick_id)]['height']
        if place_brick_ori == 1:
            width, height = height, width
        try:
            cur_voxel[new_x:new_x + height, new_y:new_y + width, new_z] = 1
            assembled_task_graph[len(assembled_task_graph) + 1] = {"x": new_x, "y": new_y, "z": new_z + 1, "brick_id": place_brick_id, "ori": place_brick_ori}
            inventory[place_brick_id] -= 1
        except Exception as e:
            print("Simulate failed!")
            pass
        current_step += 1

        new_voxel_state = np.copy(cur_state["voxel_state"])
        new_voxel_state[0, :, :, :] = cur_voxel

        new_state = {"voxel_state":new_voxel_state,
                     "inventory_state":inventory,
                     "cur_step": current_step,
                     "assembled_task_graph":assembled_task_graph,
                     "reward":0,
                     "done":0,
                     "pre_action":new_action,
                     "truncate":0,
                     "stability":cur_state["stability"]}
        
        reward, hard_violation = self.calculate_reward(cur_action, new_state)
        done = np.all(target_voxel[target_voxel == 1] == cur_voxel[target_voxel == 1])
        truncate = False
        
        # Hard constraints violated. Terminate
        if(hard_violation):
            done = True
            truncate = True
            reward = -self.big_reward
        elif current_step > self.max_steps or (not done and np.all(self.mask == 0)): # no valid actions or exceed max steps
            truncate = True
            done = True
            reward = -self.big_reward
        # Finished game
        elif done:
            done = True
            truncate = False
            reward += self.big_reward
        
        new_state["reward"] = reward
        new_state["done"] = done
        new_state["truncate"] = truncate
        return new_state

    def step(self, action_idx):
        new_state = self.simulate(self.full_state, action_idx)
        self.full_state = new_state
        inventory_state = np.copy(self.full_state["inventory_state"])
        inventory_state = np.clip(inventory_state, 0, 1)
        self.state = {"voxel_state":np.copy(self.full_state["voxel_state"]),
                      "inventory_state":np.asarray(inventory_state, dtype=np.uint8),
                      }
        return self.state, self.full_state["reward"], bool(self.full_state["done"]), bool(self.full_state["truncate"]), {"is_success":self.full_state["done"] and not self.full_state["truncate"]}

    def calculate_reward(self, cur_action, cur_state):
        total_reward = 0.0
        hard_violation = 0

        cur_voxel = cur_state["voxel_state"][0, :, :, :]
        target_voxel = cur_state["voxel_state"][1, :, :, :]
        
        cur_x, cur_y, cur_z, brick_id, ori = cur_action
        width = self.lego_lib[str(brick_id)]['width']
        height = self.lego_lib[str(brick_id)]['height']
        if ori == 1: 
            width, height = height, width

        # Count the number of voxels correctly placed
        num_on_target = self.num_vox_in_target(cur_x, cur_y, cur_z, height, width, target_voxel)
        total_reward = num_on_target / np.sum(target_voxel)
        return total_reward, hard_violation

    def num_vox_in_target(self, x, y, z, height, width, target_voxel):
        count = 0.0
        for i in range(x, x + height):
            for j in range(y, y + width):
                try:
                    if target_voxel[i, j, z] == 1:
                        count += 1.0
                except Exception as e:
                    continue
        return count

    def structure_stable(self, assembled_graph):
        analysis_score, num_vars, num_constr, total_t, solve_t = stability_score(assembled_graph, self.lego_lib, 
                                                                                 world_dim_ovr=[self.X_SIZE, self.Y_SIZE, self.Z_SIZE], 
                                                                                 config_fname="./StableLego/config.json")
        violation = analysis_score[analysis_score > 0.99]
        return len(violation) <= 0


    def visualize(self, dimension=None, save_to=""):
        if(dimension is None):
            dimension = [self.X_SIZE, self.Y_SIZE, self.Z_SIZE]        
        cur_voxel = self.state["voxel_state"][0, :, :, :]
        target_voxel = self.state["voxel_state"][1, :, :, :]
        self.visualize_current_state(cur_voxel, target_voxel, dimension=dimension, save_to=save_to)

    def visualize_current_state(self, cur_voxel, target_voxel, dimension=None, save_to=""):
        if(dimension is None):
            dimension = [self.X_SIZE, self.Y_SIZE, self.Z_SIZE]   
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def close(event):
            if event.key == 'q':
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', close)

        def add_voxel(x, y, z, color, alpha=0.8):
            ax.bar3d(x, y, z, 1, 1, 1, color, alpha=alpha)

        # Visualize target voxels
        target_voxel_indices = np.argwhere(target_voxel == 1)
        for index in target_voxel_indices:
            if(index[0] < dimension[0] and index[1] < dimension[1] and index[2] < dimension[2]):
                add_voxel(index[0], index[1], index[2], 'black')

        # Visualize current voxels
        current_voxel_indices = np.argwhere(cur_voxel == 1)
        for index in current_voxel_indices:
            if(index[0] < dimension[0] and index[1] < dimension[1] and index[2] < dimension[2]):
                add_voxel(index[0], index[1], index[2], 'red', alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Current and Target Voxel Visualization')
        ax.set_xlim((0, dimension[0]))
        ax.set_ylim((0, dimension[1]))
        ax.set_zlim((0, dimension[2]))
        ax.set_axis_off()
        ax.view_init(elev=30, azim=-60)

        if(save_to != ""):
            plt.savefig(save_to)
            plt.close(fig)
        else:
            plt.show()




