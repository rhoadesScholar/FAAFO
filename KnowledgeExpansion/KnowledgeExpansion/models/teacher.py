# Defines the teacher model and hyperparameters
# %%
import monai

teacher = monai.networks.nets.Critic(
    in_shape=(2, 224, 224), channels=(16, 32, 64, 128), strides=(2, 2, 2, 2)
)

# %%
