import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mri_data import SliceData
from model import SubSamplingModel
import utils
from SSIM import ssim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_complex_kspace_tensor(k_space):
    return torch.log(torch.sqrt(k_space[:, :, 0] ** 2 + k_space[:, :, 1] ** 2) + 1e-9)


def print_complex_image_tensor(image):
    return torch.sqrt(image[:, :, 0] ** 2 + image[:, :, 1] ** 2)


def data_transform(k_space, mask, target, attrs, f_name, slice):
    full_k_space = utils.to_tensor(k_space)

    full_image = utils.ifft2(full_k_space)

    target = utils.to_tensor(target)

    cropped_image = utils.complex_center_crop(full_image, (320, 320))

    cropped_k_space = utils.fft2(cropped_image)

    return cropped_k_space, cropped_image, slice, target


def train_model(network, train_data, number_of_epochs, learning_rate=0.05, learning_rate_decay=True):
    network = network.to(device)
    # define loos and optimizer
    optimizer = optim.Adam(network.parameters(), learning_rate)
    print('Starting Training')
    start_time = time.time()
    print("train_data len is: " + str(len(train_data)))
    for epoch_number in range(number_of_epochs):
        running_time = time.time()
        running_loss = 0
        for iter, data in enumerate(train_data):
            cropped_k_space, cropped_image, slice, target = data
            # Add channel dimension:
            cropped_k_space = cropped_k_space.unsqueeze(1).to(device)

            cropped_k_space = cropped_k_space.to(device)
            target = target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            output = network(cropped_k_space)
            loss = ssim(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            if iter % 30 == 0:
                print("Iter number is: " + str(iter))
            # print statistics
            running_loss += loss.item()
        print('running_loss(l1) = ' + str(running_loss))
        print('Epoch time: ' + str(time.time() - running_time))
        # the decaying of the learning rate the closer
        if epoch_number % 15 == 14 and learning_rate_decay:
            learning_rate *= 0.5
            optimizer = optim.Adam(network.parameters(), learning_rate)
    print('Overall time: ' + str(time.time() - start_time))
    print('Finished Training')


def test_model(model, data, limit=10, seed=0):
    # the limit makes sure the amount of picture will not be too large
    print('Starting Testing')
    # make sure to pick the same pictures every time
    np.random.seed(seed)
    i = 0
    for cropped_k_space, cropped_image, slice, target in data:
        if np.random.randn() >= 0.8:
            i += 1
            cropped_k_space = cropped_k_space.to(device)
            plt.figure()
            image = model(cropped_k_space.unsqueeze(1)).detach().cpu().squeeze()
            plt.imshow(image[0], cmap='gray')
            plt.title('the model output')
            plt.figure()
            plt.imshow(target[0].squeeze().detach().cpu(), cmap='gray')
            plt.title('target image')
            plt.show()
            if i == limit:
                break

def print_trajectory(model):
    plt.imshow(model.return_trajectory_matrix(),cmap='gray')
    plt.title('the trajectory found')
    plt.show()

def load_data():
    data_path = 'singlecoil_val'
    dataset = SliceData(
        root=data_path,
        transform=data_transform,
        challenge='singlecoil', sample_rate=1
    )
    data = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return data


def main():
    data = load_data()
    model = SubSamplingModel(decimation_rate=4, resolution=320, trajectory_learning=True)
    model = model.to(device)
    if len(data) <= 0:
        print('Dataloader failed')
        return
    else:
        train_model(model, data, 1)
        test_model(model, data)

    # path = './network_30_epochs.pth'
    # model.load_state_dict(torch.load(path))

    path = './network_1_epochs.pth'
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    # path = './network_5_epochs.pth'
    # model = SubSamplingModel(decimation_rate=1, resolution=320, trajectory_learning=True)
    # model = model.to(device)
    # model.load_state_dict(torch.load(path))
    # model.eval()
    # print_trajectory(model)
    main()
