import matplotlib.pyplot as plt
import numpy
import torch

import pytorch_nufft.interp as interp
from fastMRI.data import transforms
from pytorch_nufft import util


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)


def nufft(input, coord, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    os_shape = _get_oversamp_shape(input.shape, ndim, oversamp)

    output = input.clone()

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)

    # Zero-pad
    output = output / util.prod(input.shape[-ndim:]) ** 0.5
    output = util.resize(output, os_shape, device=device)

    # FFT
    output = transforms.rfft2(output)

    # Interpolate
    coord = _scale_coord(coord, input.shape, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.interpolate(output, width, kernel, coord, device)

    return output


def nufft_adjoint(input, coord, oshape, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = numpy.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = interp.gridding(input, os_shape, width, kernel, coord, device)

    # IFFT
    output = output.permute(0, 2, 3, 1)
    # plt.figure()
    # plt.imshow(print_complex_kspace_tensor(output[0].detach().cpu()), cmap='gray')
    # plt.show()
    # output = transforms.ifft2(output)

    # Crop
    output = output.permute(0, 3, 1, 2)
    output = util.resize(output, oshape)
    output *= util.prod(os_shape[-ndim:]) / util.prod(oshape[-ndim:]) ** 0.5

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)
    return output.permute(0, 2, 3, 1)


def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    x = torch.arange(n, dtype=dtype) / n
    kernel = 1 / width * torch.tensor(numpy.i0(beta * (1 - x ** 2) ** 0.5), dtype=dtype)
    return kernel.to(device)


def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord


def _get_ugly_number(n):
    if n <= 1:
        return n

    ugly_nums = [1]
    i2, i3, i5 = 0, 0, 0
    while (True):

        ugly_num = min(ugly_nums[i2] * 2,
                       ugly_nums[i3] * 3,
                       ugly_nums[i5] * 5)

        if ugly_num >= n:
            return ugly_num

        ugly_nums.append(ugly_num)
        if ugly_num == ugly_nums[i2] * 2:
            i2 += 1
        elif ugly_num == ugly_nums[i3] * 3:
            i3 += 1
        elif ugly_num == ugly_nums[i5] * 5:
            i5 += 1


def _get_oversamp_shape(shape, ndim, oversamp):
    return list(shape)[:-ndim] + [_get_ugly_number(oversamp * i)
                                  for i in shape[-ndim:]]


def _apodize(input, ndim, oversamp, width, beta, device):
    output = input
    for a in range(-ndim, 0):
        i = output.shape[a]
        os_i = _get_ugly_number(oversamp * i)
        idx = torch.arange(i, dtype=output.dtype, device=device)

        # Calculate apodization
        apod = (beta ** 2 - (numpy.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1))

    return output
