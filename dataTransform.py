from data import transforms


class DataTransform:
    def __init__(self, resolution) -> None:
        """
        Args:
            resolution (int): Resolution of the image. (320 for knee, 384 for brain)
        """
        self.resolution = resolution

    def __call__(self, k_space, mask, target, attrs, f_name, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                k_space (torch.Tensor): k-space(resolution x resolution x 2)
                target (torch.Tensor): Target image converted to a torch Tensor.
                fname (str): File name
                slice (int): Serial number of the slice.
        """
        k_space = transforms.to_tensor(k_space)
        full_image = transforms.ifft2(k_space)
        cropped_image = transforms.complex_center_crop(full_image, (self.resolution, self.resolution))
        k_space = transforms.fft2(cropped_image)
        # Normalize input
        cropped_image, mean, std = transforms.normalize_instance(cropped_image, eps=1e-11)
        cropped_image = cropped_image.clamp(-6, 6)
        # Normalize target
        target = transforms.to_tensor(target)
        target = transforms.center_crop(target, (self.resolution, self.resolution))
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return k_space, target, f_name, slice