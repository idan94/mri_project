import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from data import transforms
from data.mri_data import SliceData
from model import SubSamplingModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)


def data_transform(k_space, mask, target, attrs, f_name, slice):
    # print(attrs)
    full_k_space = transforms.to_tensor(k_space)
    narrowed_k_space = torch.narrow(full_k_space, 1, attrs['padding_left'],
                                    attrs['padding_right'] - attrs['padding_left'])
    full_image = transforms.ifft2(full_k_space)

    # full_image, mean, std = transforms.normalize_instance(full_image, eps=1e-11)
    # full_image = full_image.clamp(-6, 6)

    target = transforms.to_tensor(target)
    # target = transforms.normalize(target, mean, std, eps=1e-11)
    # target = target.clamp(-6, 6)

    cropped_image = transforms.complex_center_crop(full_image, (320, 320))

    cropped_k_space = transforms.fft2(cropped_image)

    return cropped_k_space, full_image, cropped_image, slice, target


def train(train_data, network, number_of_epochs):
    network = network.to(device)
    # define loos and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(network.parameters())
    print('Starting Training')
    start_time = time.time()
    print("train_data len is: " + str(len(train_data)))
    for epoch_number in range(number_of_epochs):
        running_time = time.time()
        running_loss = 0
        for iter, data in enumerate(train_data):
            cropped_k_space, full_image, cropped_image, slice, target = data
            # Add channel dimension:
            cropped_k_space = cropped_k_space.unsqueeze(1).to(device)

            cropped_k_space = cropped_k_space.to(device)
            target = target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = network(cropped_k_space)
            loss = loss_function(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("Iter number is: " + str(iter))
            # print statistics
            running_loss += loss.item()
        print('running_loss = ' + str(running_loss))
        print('Epoch time: ' + str(time.time() - running_time))
    print('Overall time: ' + str(time.time() - start_time))
    print('Finished Training')


def main():
    '''
    network = SubSamplingModel(4, 320, True)
    path = './network.pth'
    network.load_state_dict(torch.load(path))
    network = network.to(device)
    data_path = 'singlecoil_val'
    dataset = SliceData(
        root=data_path,
        transform=data_transform,
        challenge='singlecoil', sample_rate=1
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    for iter, data in enumerate(train_loader):
        cropped_k_space, full_image, cropped_image, slice, target = data
        if slice[2] == 15:
            plt.figure()
            plt.imshow(target[2], cmap='gray')
            plt.title('image')
            plt.show()
            input_to_network = cropped_k_space.unsqueeze(1).to(device)
            output_image = network(input_to_network)
            output_image = output_image.squeeze().detach().cpu()
            plt.figure()
            plt.imshow(output_image[2], cmap='gray')
            plt.title('Our network on image')
            plt.show()
            a = 5

    exit()
    '''
    print('Starting')
    data_path = 'singlecoil_val'
    dataset = SliceData(
        root=data_path,
        transform=data_transform,
        challenge='singlecoil', sample_rate=1
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    network = SubSamplingModel(4, 320, True)
    train(train_loader, network, 3)
    path = './network.pth'
    torch.save(network.state_dict(), path)


if __name__ == '__main__':
    main()
