import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data import get_data
import matplotlib.pyplot as plt
from model_class import Model
from datetime import datetime


def calculate_accuracy(dataloader, model):
    """
    :param dataloader: dataloader you want to measure the accuracy on
    :param model: Your model
    :return: accuracy as an int
    """

    model.eval()
    correct_images = 0
    total_images = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to("cuda")
            targets = batch["label"].to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(targets, outputs)
            total_images += targets.size(0)
            correct_images += (predicted == targets).sum().item()

        # print(total_images, correct_images)
        acc = 100 * correct_images // total_images
        return acc


def train(model, epochs, train_loader, test_loader, criterion, optimizer, device):
    print(f"Started training at {datetime.now()}")
    losses = []
    accs = []

    for epoch in range(epochs):
        running_loss = 0.0
        model = model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            out = model(images)

            loss = criterion(out, targets)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        acc = calculate_accuracy(test_loader, model)
        loss = running_loss / len(train_loader)
        losses.append(loss)
        accs.append(acc)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}]: loss: {loss}, train acc {acc}%, Time: {datetime.now()}")

    return accs, losses


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, test_ds = get_data()

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(train_ds, batch_size=32)

    model = Model().to(device)
    epochs = 30

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.03)

    accs, losses = train(model, epochs, train_loader, test_loader, criterion, optimizer, device)
    torch.save(model.state_dict(), "./models/emotion_detection_model.pt")

    plt.plot(accs, label="Accuracy")
    plt.plot(losses, label="Loss")
    plt.legend()
    plt.show()
