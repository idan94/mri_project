import torch
from torch.utils.data import DataLoader

from fastMRI.data import transforms
from fastMRI.data.mri_data import SliceData
from model import SubSamplingLayer

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


'''

האינפוט לשכבת דגימה צריך להיות מהצורה:
(batch_size, num_channels=1,res,res,2) לדוגמא (16,1,320,320,2)
אחרי הפקודת permute צריך להיות (16,2,320,320)
אחרי הbilinear interpolation צריך להיות (16,2,10240) המימד האחרון הוא כמות הדגימות 
ואחרי הnufft_adjoint חוזר לממדים של תמונה (16,320,320)
ואז רק מוסיף את המימד של הchannel ומחזיר (16,1,320,320)


'''


def main():
    print('Starting')
    dataset = SliceData(
        root='singlecoil_val',
        transform=data_transform,
        challenge='singlecoil', sample_rate=1
    )
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    network = SubSamplingLayer(4, 320, True)

    for iter, data in enumerate(train_loader):

        cropped_k_space, full_image, cropped_image, slice, target = data
        if slice[0] > 8:
            # Add channel dimension:
            cropped_k_space = cropped_k_space.unsqueeze(1).to(device)
            cropped_image = cropped_image.to(device)

            output = network(cropped_k_space)
            a = 5

    # for cropped_k_space, full_image, cropped_image, slice, target in dataset:
    #     if slice == 16:
    #         # print('full_k_space shape:' + str(full_k_space.shape))
    #         print('full_image shape:' + str(full_image.shape))
    #         print('cropped_image shape:' + str(cropped_image.shape))
    #         print('cropped_k_space shape:' + str(cropped_k_space.shape))
    #         # plt.figure()
    #         # plt.subplot(2, 2, 1)
    #         # plt.imshow(print_complex_kspace_tensor(narrowed_k_space), cmap='gray')
    #         # plt.subplot(2, 2, 2)
    #         # plt.imshow(print_complex_image_tensor(cropped_image), cmap='gray')
    #         # plt.subplot(2, 2, 3)
    #         # plt.imshow(print_complex_image_tensor(full_image), cmap='gray')
    #         plt.figure()
    #         plt.subplot(2, 2, 1)
    #         plt.imshow(print_complex_image_tensor(cropped_image), cmap='gray')
    #         plt.subplot(2, 2, 2)
    #         plt.imshow(target, cmap='gray')
    #         plt.subplot(2, 2, 3)
    #         plt.imshow(print_complex_kspace_tensor(cropped_k_space), cmap='gray')
    #         plt.show()
    #         print(torch.dist(torch.tensor(target), print_complex_image_tensor(cropped_image)))
    #         print("Done slice")
    #         # a += 1
    #         # if a == 4:
    #         #     exit()
    #         input = cropped_k_space.unsqueeze(1).unsqueeze(1).to(device)
    #         input.to(device)
    #         like_forward(input, 320, 1)
    #         exit()


if __name__ == '__main__':
    main()
