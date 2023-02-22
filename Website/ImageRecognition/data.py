import torchvision.transforms as tr


from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor, Compose
from torchvision import datasets

_root = r'ImageRecognition/Fruits and Vegetables Image Recognition Dataset/'
_train_images = _root + 'train/'
_test_images = _root + 'test/'


BATCH_SIZE = 20
# RECALCULATE FOR A NEW DATASET
MEAN = 0.4950
STD = 0.3253

Transforms = {
    'train': Compose([
        tr.Resize(256),
        tr.CenterCrop((224, 224)),
        tr.RandomHorizontalFlip(),
        tr.RandomRotation(10),
        ToTensor(),
        tr.Normalize(MEAN, STD)
    ]),
    'test': Compose([
        tr.Resize(256),
        tr.CenterCrop((224, 224)),
        ToTensor(),
        tr.Normalize(MEAN, STD)
    ])
}

train_dataset = datasets.ImageFolder(_train_images, transform=Transforms['train'])
test_dataset = datasets.ImageFolder(_test_images, transform=Transforms['test'])

train_dl = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

if __name__ == '__main__':
    for imgs, labels in train_dl:

        break
