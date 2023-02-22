import torch
from torch.utils.data import DataLoader
from .data import train_dl


def get_mean_and_std(dl: DataLoader):
    mean, std, counter = 0, 0, 0

    for imgs, _ in dl:
        imgs = imgs.to('cuda')

        mean += torch.mean(imgs)
        std += torch.std(imgs)
        counter += 1

    return mean / counter, std / counter


if __name__ == '__main__':
    print(get_mean_and_std(train_dl))
