import pathlib
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from common.args import Args
from data.mri_data import SliceData
from dataTransform import DataTransform
from model import SubSamplingModel
from trajectory_initiations import to_trajectory_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = Args().parse_args()
    train_data, display_data = load_data(args)
    model = SubSamplingModel(
        decimation_rate=args.decimation_rate,
        resolution=args.resolution,
        trajectory_learning=True,
        subsampling_trajectory=args.subsampling_init,
        spiral_density=args.spiral_density,
        unet_chans=args.unet_chans,
        unet_num_pool_layers=args.unet_num_pool_layers,
        unet_drop_prob=args.unet_drop_prob
    )
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
    name = 'model.pt'
    torch.save(model.state_dict(), name)


def train_model(model, train_data, display_data, args):
    model = model.to(device)

    # Define loss and optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)

    print('Starting Training')
    start_time = time.time()
    over_all_running_time = 0
    loss_fn = get_loss_fn(args)
    print("train_data len is: " + str(len(train_data)))
    for epoch_number in range(args.num_epochs):
        running_time = time.time()
        running_loss = 0
        for i, data in enumerate(train_data):
            break
            k_space, target, f_name, slice = data
            # Add channel dimension:
            k_space = k_space.unsqueeze(1).to(device)

            k_space = k_space.to(device)
            target = target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(k_space)
            output = output.to(device)
            loss = loss_fn(output, target.unsqueeze(1))
            loss.backward()

            optimizer.step()
            # print statistics
            running_loss += loss.item()
            # print(str(iter))

        print('running_loss(' + str(args.loss_fn) + ') = ' + str(running_loss))
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


def load_data(args):
    dataset = SliceData(
        root=args.data_path,
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=args.sample_rate
    )
    train_data = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    sample_rate_for_display = (args.sample_rate / (len(train_data)) * args.batch_size) * args.display_images
    display_dataset = SliceData(
        root=args.data_path,
        transform=DataTransform(resolution=args.resolution),
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
                if torch.cuda.device_count() > 1:
                    corrupted = model.module.sub_sampling_layer(k_space)
                    trajectory = model.module.sub_sampling_layer.trajectory
                else:
                    corrupted = model.sub_sampling_layer(k_space)
                    trajectory = model.sub_sampling_layer.trajectory
                trajectory = torch.tensor(to_trajectory_image(args.resolution, trajectory.cpu().detach().numpy()))
                save_image(trajectory, 'Trajectory')
                save_image(output, 'Reconstruction')
                corrupted = torch.sqrt(corrupted[..., 0] ** 2 + corrupted[..., 1] ** 2)
                save_image(corrupted, 'Corrupted')
                save_image(torch.abs(target - output), 'Error')
            break


def get_loss_fn(args):
    if args.loss_fn == 'L1':
        return nn.L1Loss()
    if args.loss_fn == 'MSE':
        return nn.MSELoss()
    if args.loss_fn == 'CEL':
        return nn.CrossEntropyLoss()
    if args.loss_fn == 'KLD':
        return nn.KLDivLoss()


if __name__ == '__main__':
    main()
