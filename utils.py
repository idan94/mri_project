import torch
import numpy as np

def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def prod(shape):
    """Computes product of shape.
    Args:
        shape (tuple or list): shape.
    Returns:
        Product.
    """
    return np.prod(shape)

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def _expand_shapes(*shapes):

    shapes = [list(shape) for shape in shapes]
    max_ndim = max(len(shape) for shape in shapes)
    shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                  for shape in shapes]

    return tuple(shapes_exp)

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

def resize(input, oshape, ishift=None, oshift=None,device='cuda'):
    ishape_exp, oshape_exp = _expand_shapes(input.shape, oshape)

    if ishape_exp == oshape_exp:
        return input.reshape(oshape)

    if ishift is None:
        ishift = [max(i // 2 - o // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    if oshift is None:
        oshift = [max(o // 2 - i // 2, 0)
                  for i, o in zip(ishape_exp, oshape_exp)]

    copy_shape = [min(i - si, o - so) for i, si, o,
                  so in zip(ishape_exp, ishift, oshape_exp, oshift)]
    islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
    oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

    output = torch.zeros(oshape_exp, dtype=input.dtype, device=device)
    input = input.reshape(ishape_exp)
    output[oslice] = input[islice]

    return output.reshape(oshape)

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def lin_interpolate(kernel, x):
    mask = torch.lt(x, 1).float()
    x = x * mask
    n = len(kernel)
    idx = torch.floor(x * n)
    frac = x * n - idx

    left = kernel[idx.long()]
    mask2 = torch.ne(idx, n - 1).float()
    idx = idx * mask2
    right = kernel[idx.long() + 1]
    output = (1.0 - frac) * left + frac * right
    return output * mask * mask2

def _scale_coord(coord, shape, oversamp, device):
    ndim = coord.shape[-1]
    scale = torch.tensor(
        [_get_ugly_number(oversamp * i) / i for i in shape[-ndim:]], device=device)
    shift = torch.tensor(
        [_get_ugly_number(oversamp * i) // 2 for i in shape[-ndim:]], device=device, dtype=torch.float32)

    coord = scale * coord + shift

    return coord


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

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
        apod = (beta ** 2 - (np.pi * width * (idx - i // 2) / os_i) ** 2) ** 0.5
        apod = apod / torch.sinh(apod)
        output = output * apod.reshape([i] + [1] * (-a - 1))

    return output


def bilinear_interpolate_torch_gridsample(input, coord):
    coord = coord.unsqueeze(0).unsqueeze(0)
    tmp = torch.zeros_like(coord)
    tmp[:, :, :, 0] = (
            (coord[:, :, :, 1] + input.shape[2] / 2) / (input.shape[2] - 1))  # normalize to between  -1 and 1
    tmp[:, :, :, 1] = (
            (coord[:, :, :, 0] + input.shape[2] / 2) / (input.shape[2] - 1))  # normalize to between  -1 and 1
    tmp = tmp * 2 - 1  # normalize to between -1 and 1
    tmp = tmp.expand(input.shape[0], -1, -1, -1)
    return torch.nn.functional.grid_sample(input=input, grid=tmp, mode='bilinear', padding_mode='zeros').squeeze(2)

def _get_kaiser_bessel_kernel(n, width, beta, dtype, device):
    x = torch.arange(n, dtype=dtype) / n
    kernel = 1 / width * torch.tensor(np.i0(beta * (1 - x ** 2) ** 0.5), dtype=dtype)
    return kernel.to(device)

def nufft_adjoint(input, coord, oshape, oversamp=1.25, width=4.0, n=128, device='cuda'):
    ndim = coord.shape[-1]
    beta = np.pi * (((width / oversamp) * (oversamp - 0.5)) ** 2 - 0.8) ** 0.5
    oshape = list(oshape)

    os_shape = _get_oversamp_shape(oshape, ndim, oversamp)

    # Gridding
    coord = _scale_coord(coord, oshape, oversamp, device)
    kernel = _get_kaiser_bessel_kernel(n, width, beta, coord.dtype, device)
    output = gridding(input, os_shape, width, kernel, coord, device)

    # IFFT
    output = output.permute(0, 2, 3, 1)

    # plt.figure()
    # plt.imshow(print_complex_kspace_tensor(output[0].detach().cpu()), cmap='gray')
    # plt.show()

    output = ifft2(output)

    # plt.figure()
    # plt.imshow(print_complex_image_tensor(output[0].detach().cpu()), cmap='gray')
    # plt.show()

    # Crop
    output = output.permute(0, 3, 1, 2)
    output = resize(output, oshape, device=device)
    output *= prod(os_shape[-ndim:]) / prod(oshape[-ndim:]) ** 0.5

    # Apodize
    output = _apodize(output, ndim, oversamp, width, beta, device)
    return output.permute(0, 2, 3, 1)


def gridding(input, shape, width, kernel, coord, device):
    ndim = coord.shape[-1]

    batch_shape = shape[:-ndim]
    batch_size = prod(batch_shape)

    pts_shape = coord.shape[:-1]
    npts = prod(pts_shape)

    input = input.reshape([batch_size, npts])
    coord = coord.reshape([npts, ndim])
    output = torch.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype, device=device)

    output = _gridding2(output, input, width, kernel, coord)

    return output.reshape(shape)


def _gridding2(output, input, width, kernel, coord):
    batch_size, ny, nx = output.shape

    kx, ky = coord[:, -1], coord[:, -2]

    x0, y0 = (torch.ceil(kx - width / 2),
              torch.ceil(ky - width / 2))

    for y in range(int(width) + 1):
        wy = lin_interpolate(kernel, torch.abs(y0 + y - ky) / (width / 2))

        for x in range(int(width) + 1):
            w = wy * lin_interpolate(kernel, torch.abs(x0 + x - kx) / (width / 2))

            yy = torch.fmod(y0 + y, ny).long()
            xx = torch.fmod(x0 + x, nx).long()
            output[:, yy, xx] = output[:, yy, xx] + w * input[:, :]

    return output