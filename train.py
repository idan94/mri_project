import pathlib
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional
from torch.utils.data import DataLoader

from common.args import Args
from data import transforms
from data.mri_data import SliceData
from model import SubSamplingModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)


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
        target = transforms.to_tensor(target)
        return k_space, target, f_name, slice


def save_model(args, output_dir, epoch, model, optimizer):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'output_dir': output_dir,
        },
        f=str(output_dir) + '/model.pt'
    )


def visualize(args, epoch, model, data_loader):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        args.writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            k_space, target, f_name, slice = data
            k_space = k_space.unsqueeze(1).to(device)
            target = target.unsqueeze(1).to(device)
            save_image(target, 'Target')
            if epoch != 0:
                output = model(k_space.clone())
                corrupted = model.sub_sampling_layer(k_space)

                save_image(output, 'Reconstruction')
                corrupted = torch.sqrt(corrupted[..., 0] ** 2 + corrupted[..., 1] ** 2)
                save_image(corrupted, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break


def train_model(model, train_data, display_data, args):
    model = model.to(device)

    # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)

    print('Starting Training')
    start_time = time.time()
    over_all_running_time = 0
    print("train_data len is: " + str(len(train_data)))
    for epoch_number in range(args.num_epochs):
        running_time = time.time()
        running_loss = 0
        for i, data in enumerate(train_data):
            k_space, target, f_name, slice = data
            # Add channel dimension:
            k_space = k_space.unsqueeze(1).to(device)

            k_space = k_space.to(device)
            target = target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(k_space)
            loss = functional.l1_loss(output.squeeze(), target)
            loss.backward()

            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print(str(iter))

        print('running_loss(L1) = ' + str(running_loss))
        print('Epoch time: ' + str(time.time() - running_time))
        over_all_running_time += (time.time() - running_time)
        visualize(args, epoch_number + 1, model, display_data)
        torch.save(model.state_dict, args.output_dir + '/model_test.pth')

    # save_model(args, args.output_dir, epoch_number, model, optimizer)

    print('Overall run time: ' + str(time.time() - start_time))
    print('Overall train time: ' + str(over_all_running_time))
    print(args.test_name)
    print('Finished Training')
    args.writer.close()


def test_model(model, data, args):
    print('Starting Testing')
    with torch.no_grad():
        for k_space, target, f_name, slice in data:
            for i in range(args.display_images):
                k_space = k_space.to(device)

                plt.figure()
                image = model(k_space.unsqueeze(1)).detach().cpu().squeeze()
                plt.imshow(image[i], cmap='gray')
                plt.title('the model output')

                plt.figure()
                plt.imshow(target[i].squeeze().detach().cpu(), cmap='gray')
                plt.title('target image')
                plt.show()


def print_trajectory(model):
    plt.imshow(model.return_trajectory_matrix(), cmap='gray')
    plt.title('the trajectory found')
    plt.show()


def load_data(args):
    dataset = SliceData(
        root=args.data_path,
        transform=DataTransform(resolution=320),
        challenge=args.challenge, sample_rate=args.sample_rate
    )

    train_data = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    sample_rate_for_display = (args.sample_rate / (len(train_data)) * args.batch_size) * args.display_images
    display_dataset = SliceData(
        root=args.data_path,
        transform=DataTransform(resolution=320),
        challenge=args.challenge, sample_rate=sample_rate_for_display
    )
    display_data = DataLoader(
        dataset=display_dataset,
        batch_size=args.display_images,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_data, display_data


def main():
    args = Args().parse_args()
    train_data, display_data = load_data(args)
    model = SubSamplingModel(decimation_rate=args.decimation_rate, resolution=args.resolution, trajectory_learning=True)
    # Multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    args.output_dir = 'outputs/' + args.output_dir
    args.writer = SummaryWriter(log_dir=args.output_dir)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output_dir + '/args.txt', "w") as text_file:
        print(vars(args), file=text_file)
    train_model(model, train_data, display_data, args)


if __name__ == '__main__':
    main()
