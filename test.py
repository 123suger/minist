import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt


def image_show(images):
    images = images.numpy()
    images = images.transpose((1, 2, 0))
    print(images.shape)
    plt.imshow(images)
    plt.show()


def main():
    batch_size = 32
    # train_dataset = datasets.MNIST(root='./datasets', train=False, download=False,
    #                                transform=transforms.ToTensor())
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    train_db = datasets.MNIST('mnist', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),    # 转换数据类型为张量
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size, shuffle=True)
    device = torch.device('cuda:0')
    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    #     inputs = inputs.to(device)
    #     print(inputs.shape)
    inputs, targets = next(iter(train_loader))
    print(inputs.shape)
    print(targets.shape)
    print(targets)
    images = torchvision.utils.make_grid(inputs)    # 进行读入数据的拼接组合，生成的量依旧为张量
    print(f'images.shape:{images.shape}')
    image_show(images)


if __name__ == '__main__':
    main()