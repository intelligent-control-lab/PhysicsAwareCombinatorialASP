from utils_gym import *

class Policy(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, features_dim: int = 2048):
        super().__init__(observation_space, features_dim)
        voxel_state = observation_space["voxel_state"]

        self.cnn3d = nn.Sequential(
            nn.Conv3d(2, 8, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.Conv3d(8, 32, kernel_size=5, stride=2),
            nn.Tanh(),
            nn.Flatten(),
        )
        print("Feature dim:", features_dim)
        # Compute the output size of Flatten layer dynamically
        with torch.no_grad():
            sample_input = torch.as_tensor(voxel_state.sample()[None]).float()
            n_flatten = self.cnn3d(sample_input).shape[1]
        print("CNN 3D dim: ", n_flatten)
        
        # Merge pathway
        self.merged_net = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        voxel_state = observations["voxel_state"]
        voxel_features = self.cnn3d(voxel_state)
        return self.merged_net(voxel_features)