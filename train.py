import time

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional
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

    return cropped_k_space, cropped_image, slice, target


def train_model(network, train_data, number_of_epochs):
    network = network.to(device)
    # define loos and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), 0.03)
    a = [b for b in network.parameters()]
    print(network.parameters())
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
            loss = functional.l1_loss(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            if iter % 30 == 0:
                print("Iter number is: " + str(iter))
            # print statistics
            running_loss += loss.item()
        print('running_loss(l1) = ' + str(running_loss))
        print('Epoch time: ' + str(time.time() - running_time))
    print('Overall time: ' + str(time.time() - start_time))
    print('Finished Training')


def test_model(model, data):
    print('Starting Testing')
    i = 0
    for cropped_k_space, cropped_image, slice, target in data:
        if slice[0] > 10:
            i += 1
            if i % 10 == 0:
                cropped_k_space = cropped_k_space.to(device)
                plt.figure()
                image = model(cropped_k_space.unsqueeze(1)).detach().cpu().squeeze()
                plt.imshow(image[0], cmap='gray')
                plt.title('the model output')
                plt.figure()
                plt.imshow(target[0].squeeze().detach().cpu(), cmap='gray')
                plt.title('target image')
                plt.show()
                a = 5


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

    # Print summary of the model
    # summary(model, (1, 320, 320, 2))

    if len(data) <= 0:
        print('Dataloader failed')
        return
    else:
        train_model(model, data, 5)
        test_model(model, data)
    # path = './network_30_epochs.pth'
    # model.load_state_dict(torch.load(path))

    # train(train_loader, network, 20)
    # path = './network.pth'
    # torch.save(network.state_dict(), path)


if __name__ == '__main__':
    main()
