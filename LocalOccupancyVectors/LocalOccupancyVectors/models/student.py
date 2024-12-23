# Defines the student model and hyperparameters
# %%
import monai

# student = monai.networks.nets.ViTAutoEnc(
#     1, [224, 224], 16, out_channels=1, spatial_dims=2
# )

student = lambda out_channels: monai.networks.nets.SwinUNETR(
    img_size=(224, 224), in_channels=1, out_channels=out_channels, spatial_dims=2, use_v2=True
)

# %%
