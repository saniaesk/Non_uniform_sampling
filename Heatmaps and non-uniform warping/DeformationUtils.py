from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# =========================
# Patch generator
# =========================
def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """Return NxHxWxC float/np array of overlapping patches from HxW or HxWxC image."""
    if image.ndim == 2:
        img = image[..., None]  # HxWx1
    elif image.ndim == 3:
        img = image
    else:
        raise ValueError(f"image must be HxW or HxWxC, got shape={image.shape}")

    H, W, C = img.shape
    patches = []
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patches.append(img[i:i + patch_size, j:j + patch_size, :])
    if not patches:
        return np.empty((0, patch_size, patch_size, C), dtype=img.dtype)
    return np.stack(patches, axis=0)


# =========================
# Gaussian helpers
# =========================
def makeGaussian(size: int, fwhm: float = 3.0, center: tuple[int, int] | None = None) -> np.ndarray:
    """Square 2D Gaussian kernel parameterized by FWHM."""
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / (fwhm ** 2))


def Gaussian_1d(size: int, fwhm: float) -> np.ndarray:
    """1-D Gaussian kernel (row vector) parameterized by FWHM."""
    x = np.arange(0, size, 1, float)
    x0 = size // 2
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2) / (fwhm ** 2))


def uniform_grids_1d(grid_size: int, padding_size: int):
    """Uniform destination coordinates for structure-driven grid (as in the repo)."""
    global_size = grid_size + 2 * padding_size
    uniform_x = np.zeros((1, global_size))
    uniform_y = np.zeros((global_size, 1))
    for i in range(global_size):
        uniform_x[0, i] = (i - padding_size) / (grid_size - 1.0)
        uniform_y[i, 0] = (i - padding_size) / (grid_size - 1.0)
    return uniform_x, uniform_y


def gauss_conv_1d(_input: torch.Tensor, fwhm: float, axis: str, padding_size: int = 30) -> torch.Tensor:
    """
    1D Gaussian convolution along x or y using Conv2d with weights set explicitly.
    _input: [N,1,H,W] (but H or W may be 1 after max-reduction in the caller).
    """
    gauss_size = 2 * padding_size + 1
    if axis == "x":
        # kernel shape 1xK (horizontal)
        g = torch.tensor(Gaussian_1d(gauss_size, fwhm), dtype=torch.float32)  # [K]
        conv = nn.Conv2d(1, 1, kernel_size=(1, gauss_size), bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight.data[0, 0, 0, :] = g
    elif axis == "y":
        # kernel shape Kx1 (vertical)
        g = torch.tensor(Gaussian_1d(gauss_size, fwhm), dtype=torch.float32)  # [K]
        conv = nn.Conv2d(1, 1, kernel_size=(gauss_size, 1), bias=False)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight.data[0, 0, :, 0] = g
    else:
        raise ValueError("axis must be 'x' or 'y'")
    return conv(_input)


# =========================
# Pixel-driven deformation grid
# =========================
def warped_imgs(img: torch.Tensor, heat: torch.Tensor, res: tuple[int, int], fwhm: float, scale: float):
    """
    Build pixel-driven grid and sample image (repo's 'warped_imgs').
    img, heat: HxW tensors (float32).
    Returns: (sampled_image [1,1,H,W], grid [1,H,W,2] in [-1,1])
    """
    img = img.type(torch.float32).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    heat = heat.type(torch.float32).unsqueeze(0).unsqueeze(0) # [1,1,H,W]

    n, c, h, w = heat.shape
    f = heat.reshape(n, c, -1)
    x = torch.softmax(f * scale, dim=2).view(n, c, h, w)

    grid_size = (144, 112)
    padding_size = 28
    global_size = (grid_size[0] + 2 * padding_size, grid_size[1] + 2 * padding_size)

    # 2D Gaussian conv kernel (explicit)
    g2 = torch.from_numpy(makeGaussian(2 * padding_size + 1, fwhm)).float()  # [K,K]
    conv2 = nn.Conv2d(1, 1, kernel_size=(2 * padding_size + 1, 2 * padding_size + 1), bias=False)
    with torch.no_grad():
        conv2.weight.zero_()
        conv2.weight.data[0, 0, :, :] = g2

    # P basis
    P_basis = torch.zeros(2, global_size[0], global_size[1])
    for k in range(2):
        for i in range(global_size[0]):
            for j in range(global_size[1]):
                P_basis[k, i, j] = (
                    k * (i - padding_size) / (grid_size[0] - 1.0) +
                    (1.0 - k) * (j - padding_size) / (grid_size[1] - 1.0)
                )

    x = nn.Upsample(size=grid_size, mode="bilinear")(x)
    x = nn.ReplicationPad2d(padding_size)(x)

    P = torch.zeros(1, 2, global_size[0], global_size[1], requires_grad=False)
    P[0, :, :, :] = P_basis
    P = P.expand(x.size(0), 2, global_size[0], global_size[1])

    x_cat = torch.cat((x, x), 1)
    p_filter = conv2(x)
    x_mul = (P * x_cat).view(-1, 1, global_size[0], global_size[1])
    all_filter = conv2(x_mul).view(-1, 2, grid_size[0], grid_size[1])

    x_filter = all_filter[:, 0:1, :, :] / p_filter
    y_filter = all_filter[:, 1:1+1, :, :] / p_filter

    xgrids = torch.clamp(x_filter * 2 - 1, -1, 1)
    ygrids = torch.clamp(y_filter * 2 - 1, -1, 1)

    grid = torch.cat((xgrids, ygrids), 1)                   # [1,2,h,w]
    grid = nn.Upsample(size=res, mode="bilinear")(grid)     # [1,2,H,W]
    grid = grid.transpose(1, 2).transpose(2, 3)             # [1,H,W,2]
    grid = torch.clamp(grid, -1.0, 1.0)

    x_sampled = F.grid_sample(img, grid, align_corners=False)
    return x_sampled, grid


# =========================
# Structure-driven deformation grid
# =========================
def warped_str(img: torch.Tensor, heat: torch.Tensor, input_size_net: tuple[int, int],
               fwhm: float, scale: float):
    """
    Build structure-driven grid and sample image (repo's 'warped_str').
    img, heat: HxW tensors (float32).
    Returns: (sampled_image [1,1,H,W], grid [1,H,W,2] in [-1,1])
    """
    img = img.type(torch.float32).unsqueeze(0).unsqueeze(0)    # [1,1,H,W]
    heat = heat.type(torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    heat = nn.Upsample(size=(112, 112))(heat)

    n, c, h, w = heat.shape
    f = heat.reshape(n, c, -1)
    x = torch.softmax(f * scale, dim=2).view(n, c, h, w)

    grid_size = 112
    padding_size = 30
    saliency = nn.ReplicationPad2d(padding_size)(x)

    global_size = grid_size + 2 * padding_size
    dst_x, dst_y = uniform_grids_1d(grid_size, padding_size)
    uniform_x = torch.FloatTensor(dst_x[None, None, :, :]).expand(n, -1, -1, -1)
    uniform_y = torch.FloatTensor(dst_y[None, None, :, :]).expand(n, -1, -1, -1)

    # Max-reduce and 1D smooth (explicit Gaussian convs)
    saliency_x, _ = torch.max(saliency, dim=2, keepdim=True)
    saliency_y, _ = torch.max(saliency, dim=3, keepdim=True)

    denominator_x = gauss_conv_1d(saliency_x, fwhm, axis="x")
    numerator_x   = gauss_conv_1d(saliency_x * uniform_x, fwhm, axis="x")
    src_xgrids    = numerator_x / (denominator_x + 1e-8)

    denominator_y = gauss_conv_1d(saliency_y, fwhm, axis="y")
    numerator_y   = gauss_conv_1d(saliency_y * uniform_y, fwhm, axis="y")
    src_ygrids    = numerator_y / (denominator_y + 1e-8)

    # Normalize to [-1,1] and expand to HxW
    xgrids = torch.clamp(src_xgrids * 2 - 1, -1, 1).expand(-1, -1, src_xgrids.shape[3], -1)
    ygrids = torch.clamp(src_ygrids * 2 - 1, -1, 1).expand(-1, -1, -1, src_ygrids.shape[2])

    xgrids = xgrids.view(-1, 1, grid_size, grid_size)
    ygrids = ygrids.view(-1, 1, grid_size, grid_size)
    grid   = torch.cat((xgrids, ygrids), 1)

    grid = nn.Upsample(size=input_size_net, mode="bilinear")(grid)
    grid = grid.transpose(1, 2).transpose(2, 3)
    grid = torch.clamp(grid, -1.0, 1.0)

    x_sampled = F.grid_sample(img, grid, align_corners=False)
    return x_sampled.detach(), grid.detach()


# =========================
# Convenience: blended resample
# =========================
def get_resampled_images(img: torch.Tensor, heat: torch.Tensor,
                         res: tuple[int, int], fwhm: float, scale: float, lambd: float):
    """Blend structure/pixel grids with weight lambd (0..1) and sample image."""
    img = img.type(torch.float32)
    s_img, structured_grid = warped_str(img, heat, res, fwhm, scale)
    p_img, pixel_grid      = warped_imgs(img, heat, res, fwhm, scale)
    src_grid = torch.clamp((1.0 - lambd) * structured_grid + lambd * pixel_grid, -1.0, 1.0)
    img_bchw = img.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    x_sampled = F.grid_sample(img_bchw, src_grid, align_corners=False)
    return x_sampled.detach(), src_grid.detach()
