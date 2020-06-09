import torch
import numpy as np
from torch import nn
from data.transforms import fft2 as fft, ifft2 as ifft
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.mri_data import SliceData
from dataTransform import DataTransform
from common.args import Args


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
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_data_loader, val_data_loader, display_data_loader


class subsampeling_NN(nn.Module):
    def __init__(self, resolution, max_window_width, number_of_image_convolutions, number_of_linear_layers,
                 number_of_window_convolutions,
                 number_of_conjoined_convolutions):
        super().__init__()
        self.max_window_size = max_window_width
        self.image_convolutions = nn.Sequential(
            nn.Conv2d(2, 4, 3,padding=1),
            nn.MaxPool2d(2),
            nn.Sequential(
                *[nn.Sequential(nn.Conv2d(4, 4, 3,padding=1), nn.MaxPool2d(2)) for _ in range(number_of_image_convolutions - 2)]),
            nn.Conv2d(4, 2, 3,padding=1),
        )
        number_of_feature = (resolution ** 2) // (4 ** (number_of_image_convolutions - 1))
        number_of_initial_neurons = (max_window_width ** 2) * (2 ** (number_of_linear_layers - 1))

        self.linear_layers = nn.Sequential(
            nn.Linear(number_of_feature, number_of_initial_neurons),
            nn.ReLU(),
            nn.Sequential(*[nn.Sequential(nn.Linear(number_of_initial_neurons // (2 ** length),
                                                    number_of_initial_neurons // (2 ** (length + 1)), ), nn.ReLU()) for
                            length in
                            range(number_of_linear_layers - 1)])
        )
        self.window_convolutions = nn.Sequential(
            nn.Conv2d(2, 4, 3,padding=1),
            nn.Sequential(
                *[nn.Sequential(nn.Conv2d(4, 4, 3,padding=1), nn.Tanh()) for i in range(number_of_window_convolutions - 2)]
                ),
            nn.Conv2d(4, 2, 3,padding=1)
        )

        self.final_convolutions = nn.Sequential(
            nn.Conv2d(4, 8, 3,padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 6, 3,padding=1),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3,padding=1),
            nn.Tanh(),
            nn.Conv2d(4, 1, 3,padding=1),
        )

    def forward(self, image, window_size, window_index, window_filling):
        # the center of the window is in window_index
        new_image = image.clone()
        new_image = new_image.permute(0,3,1,2)
        window = new_image[:,:,window_index[0] - window_size//2:window_index[0] + window_size//2,
        window_index[1] - window_size//2:window_index[1] + window_size//2]

        window = self.window_convolutions(window)

        new_image[:,:,window_index[0] - window_size:window_index[0] + window_size,
        window_index[1] - window_size:window_index[1] + window_size] = window_filling
        features = self.image_convolutions(new_image)
        batch_size,channels,_,_ = features.shape
        features = features.reshape(batch_size, channels,-1)
        features = self.linear_layers(features)
        features = features.reshape(batch_size,channels,self.max_window_size,self.max_window_size)
        middle = self.max_window_size // 2
        # cut the features to match the size of the window
        features = features[:, :, middle - window_size // 2:middle + window_size // 2,
                   middle - window_size // 2:middle + window_size // 2]
        output = torch.cat((features,window),dim=1)
        output = self.final_convolutions(output)
        shape = output.shape
        output = output.reshape(batch_size,-1)
        # to get the right sampling we will do something close to max-pool
        maximals = [torch.argmax(output[i,...]) for i in  range(output.shape[0])]
        masks = torch.zeros_like(output)
        for i in range(len(maximals)):
            index = maximals[i].detach().item()
            masks[i,index] = output[i,index]
        return masks



def main():
    args = Args().parse_args()
    train_data_loader, val_data_loader, display_data_loader = load_data(args)
    model = subsampeling_NN(args.resolution, 6, 3, 3, 3, 3)
    for k_space, target, f_name, slice in display_data_loader:
        window_index = (k_space.shape[1] // 2, k_space.shape[1] // 2)
        a = model(k_space,4,window_index,-1)
        a = torch.norm(a)
        a.backward()
        break




if __name__ == '__main__':
    main()
