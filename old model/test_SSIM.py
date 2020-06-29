import torch
from SSIM import ssim
from train import load_data
from common.args import Args
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = Args().parse_args()
    torch.random.manual_seed(0)
    data = load_data(args)
    images = [target.squeeze() for (k_space, target, f_name, slice) in data[1]]
    image1 = images[3]
    image_to_compare = image1.unsqueeze(0).unsqueeze(0)
    image_to_find = torch.zeros_like(image_to_compare)
    image_to_compare.requires_grad = True
    image_to_find.requires_grad = True
    optimizer = torch.optim.Adam([image_to_find],lr= 0.001)
    for i in range(100):
        optimizer.zero_grad()
        loss = 1-ssim(image_to_compare,image_to_find,window_size=11)
        loss.backward()
        optimizer.step()
        if i % 10 == 9:
            print('iteration number: '+ str(i+1))
            print('loss is: ' + str(loss.detach().item()))
    SSIM = ssim(image_to_compare, image_to_find)
    print('the SSIM between the images'+str(SSIM.item()))
    plt.figure()
    plt.imshow(image1.numpy(),cmap='gray')
    plt.title('the image')
    plt.show()

    plt.figure()
    plt.imshow(image_to_find.squeeze().detach().numpy(), cmap='gray')
    plt.title('the found image image')
    plt.show()



