# Defines the teacher model and hyperparameters
# %%
import monai
import torch

# Define the teacher model
# The teacher model is a ViT visual transformer network with an image input shape and a mask input, outputting a single scalar value


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = monai.networks.nets.Critic(
        #     in_shape=(2, 224, 224), channels=(16, 32, 64, 128), strides=(2, 2, 2, 2)
        # )
        self.model = monai.networks.nets.ViT(
            in_channels=2,
            img_size=224,
            patch_size=16,
            num_classes=1,
            spatial_dims=2,
            classification=True,
            post_activation=torch.nn.Identity(),
        )

    def forward(self, image, mask):
        # return self.model(torch.concat([image, mask], dim=1))
        out, _ = self.model(torch.concat([image, mask], dim=1))
        return out


teacher = Teacher()

# %%
