import torch
import os.path
import matplotlib.pyplot as plt

from torch import nn
from torchvision.models import resnet101

from .data import test_dl, train_dl

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = resnet101().to(dev)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(dev), labels.to(dev)

        # Compute prediction error
        pred = model(imgs)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(imgs)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            pred = model(imgs)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct * 100, round(test_loss, 5)


def _main(epochs=40):
    accuracies = []
    losses = []
    for t in range(epochs):
        try:
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dl, model, loss_fn, optimizer)
            acc, loss = test(test_dl, model, loss_fn)

            accuracies.append(acc)
            losses.append(loss)
        except KeyboardInterrupt:
            break

    if os.path.exists('model.pth'):
        choice = input('Do you want to override existing model?')
        if choice.lower() == 'y':
            torch.save(model.state_dict(), 'model.pth')

    torch.save(model.state_dict(), 'model.pth')

    plt.plot(accuracies)
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('%')
    plt.show()

    plt.plot(losses)
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    _main(60)
