import torch
import torch.nn.functional as func
from PIL import Image
from torchvision.models import resnet101

from .data import train_dataset, Transforms

CLASSES = train_dataset.classes

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

Model = resnet101().to(dev)
Model.load_state_dict(torch.load('ImageRecognition/model.pth'))


def inference(model: torch.nn.Module, img: Image) -> list[dict[str:float]]:
    Model.eval()
    """
    Function receives an image and identifies it with given model
    :param model: PyTorch model
    :param img: PIL.Image
    :return: list[dict[str:float]]
    """
    # THIS IS NEEDED WHENEVER THE IMAGE HAS LESS THAN 3 CHANNELS
    img = img.convert('RGB')
    
    tensor = Transforms['test'](img).float()
    tensor = tensor.unsqueeze(0).cuda()

    logits = model(tensor)
    probs = func.softmax(logits, dim=1)
    conf, classes = torch.topk(probs, 5, dim=1)

    return [
        {
            'class': CLASSES[x],
            'confidence': round(float(y), 5)
        } for (x, y) in zip(classes.data[0], conf.data[0])
    ]


if __name__ == '__main__':
    print(inference(Model, Image.open(open('img.jpg', 'rb'))))
