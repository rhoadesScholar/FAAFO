# Defines the student model and hyperparameters
# %%
import monai

student = monai.networks.nets.ViTAutoEnc(
    1, [224, 224], 16, out_channels=1, spatial_dims=2
)
