# Load MONAI ViT model
# %%
import monai
import torch

_model = monai.networks.nets.ViTAutoEnc(
    in_channels=2,
    img_size=[224, 224],
    patch_size=16,
    out_channels=2,
    spatial_dims=2,
    proj_type="perceptron",
)


class FourierSpaceModel(torch.nn.Module):
    def __init__(self, model=_model, spatial_dims=2):
        super().__init__()
        self.model = model
        self.spatial_dims = spatial_dims

    def forward(self, x):
        if x.ndim == self.spatial_dims + 1:
            x = torch.stack([torch.real(x), torch.imag(x)], dim=1)
        elif x.ndim == self.spatial_dims + 2:
            x = torch.concatenate([torch.real(x), torch.imag(x)], dim=1)
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        # out = torch.complex(out[:, 0], out[:, 1])
        return out

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        self.model.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)

    def render(self, x):
        x = torch.complex(x[:, 0], x[:, 1])
        x = torch.fft.ifftshift(x)
        out = torch.fft.ifft2(x)
        return out[:, None, ...]


# %%
