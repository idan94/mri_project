import pathlib
import time
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
    # Args stuff:
    args = Args().parse_args()
    args.output_dir = 'outputs/' + args.output_dir
    writer = SummaryWriter(log_dir=args.output_dir)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(args.output_dir + '/args.txt', "w") as text_file:
        for arg in vars(args):
            print(str(arg) + ': ' + str(getattr(args, arg)), file=text_file)

    # Load data
    train_data_loader, val_data_loader, display_data_loader = load_data(args)

    # Define model:
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
    # # Multiple GPUs:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    start_epoch = 0
    # Define optimizer:
    if torch.cuda.device_count() > 1:
        sub_parameters = model.module.sub_sampling_layer.parameters()
        recon_parameters = model.module.reconstruction_model.parameters()
    else:
        sub_parameters = model.sub_sampling_layer.parameters()
        recon_parameters = model.reconstruction_model.parameters()

    optimizer = optim.Adam(
        [{'params': sub_parameters, 'lr': args.sub_lr},
         {'params': recon_parameters}]
        , args.lr)
    # Check if to resume or new train
    if args.resume is True:
        checkpoint = torch.load(pathlib.Path('outputs/' + args.checkpoint + '/model.pt'))
        old_args = checkpoint['args']
        # Check if the old and new args are matching
        assert (args.resolution == old_args.resolution)
        assert (args.challenge == old_args.challenge)
        assert (args.unet_chans == old_args.unet_chans)
        assert (args.unet_drop_prob == old_args.unet_drop_prob)
        assert (args.unet_num_pool_layers == old_args.unet_num_pool_layers)
        assert (args.decimation_rate == old_args.decimation_rate)
        # Load model
        model.load_state_dict(checkpoint['model'])
        # Load optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set epoch number
        start_epoch = checkpoint['epoch'] + 1
    # Train
    train_model(model, optimizer, train_data_loader, display_data_loader, args, writer, start_epoch)

def penalty(model,args):
    trajectory = model.get_trajectory()
    derivative = trajectory[:-1,:] - trajectory[1:,:]
    second_derivative = derivative[:-1,:] - derivative[1:,:]
    return args.penalty_weight*(torch.norm(derivative,2) + torch.norm(second_derivative,2))

def train_model(model, optimizer, train_data, display_data, args, writer, start_epoch):
    print('~~~Starting Training~~~')
    start_time = time.time()
    over_all_running_time = 0
    loss_fn = get_loss_fn(args)
    print("train_data len is: " + str(len(train_data)))
    for epoch_number in range(start_epoch, start_epoch + args.num_epochs):
        running_time = time.time()
        running_loss = 0
        if epoch_number % args.penalty_increment_iteration_number == args.penalty_increment_iteration_number -1:
            args.penalty_weight *= args.penalty_increment
        for i, data in enumerate(train_data):
            k_space, target, f_name, slice = data
            # Add channel dimension:
            k_space = k_space.unsqueeze(1).to(device)
            # Move to device
            k_space = k_space.to(device)
            target = target.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Use model
            output = model(k_space)
            # Move to device
            output = output.to(device)
            # Calculate loss
            if torch.cuda.device_count() > 1:
                loss = loss_fn(output, target.unsqueeze(1)) + penalty(model.module,args)
            else:
                loss = loss_fn(output, target.unsqueeze(1)) + penalty(model, args)
            loss.backward()
            # Make a step(update model parameters)
            optimizer.step()
            running_loss += loss.item()

        # Print training progress and save model
        status_printing = \
            'Epoch Number: ' + str(epoch_number+1) + '\n' \
            + 'Running_loss(' + str(args.loss_fn) + ') = ' + str(running_loss) + '\n' \
            + 'Epoch time: ' + str(time.time() - running_time)
        over_all_running_time += (time.time() - running_time)
        save_model(args, epoch_number+1, model, optimizer)
        visualize(args, epoch_number+1, model, display_data, writer)
        print(status_printing)

    # Print train statistics
    print('Overall run time: ' + str(time.time() - start_time))
    print('Overall train time: ' + str(over_all_running_time))
    print('~~~Finished Training~~~')
    writer.close()


def save_model(args, epoch_number, model, optimizer):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_number,
            'args': args,
        },
        pathlib.Path(args.output_dir + '/model.pt'),
    )


def load_data(args):
    train_dataset = SliceData(
        root=args.data_path + '/singlecoil_train',
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=args.sample_rate
    )
    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataset = SliceData(
        root=args.data_path + '/singlecoil_val',
        transform=DataTransform(resolution=args.resolution),
        challenge=args.challenge, sample_rate=args.sample_rate
    )
    val_data_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    display_dataset = [val_dataset[i] for i in
                       range(0, len(val_dataset), len(val_dataset) // args.display_images)]
    display_data_loader = DataLoader(
        dataset=display_dataset,
        batch_size=args.display_images,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_data_loader, val_data_loader, display_data_loader


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

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
                trajectory_image = torch.tensor(to_trajectory_image(args.resolution, trajectory.cpu().detach().numpy()))
                save_image(trajectory_image, 'Trajectory')
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
