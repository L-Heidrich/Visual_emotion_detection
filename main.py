import itertools

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data import get_data
import matplotlib.pyplot as plt
from model_class import Model
from datetime import datetime


def calculate_accuracy(dataloader, model, device):
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
            images = batch["image"].to(device)
            targets = batch["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(targets, outputs)
            total_images += targets.size(0)
            correct_images += (predicted == targets).sum().item()

        # print(total_images, correct_images)
        acc = 100 * correct_images // total_images
        return acc


def train(models_list, epochs_list, train_loader, val_loader, test_loader, criterion, learning_rates, device):
    """
    :param model: array of pytorch model classes
    :param epochs: array of epochs numbers
    :param train_loader: dataloader for training data
    :param test_loader: dataloader for test data
    :param val_loader: dataloader for validation data
    :param criterion: loss function
    :param learning_rates: learning rates to be used
    :param device: where to train the model on. cuda or cpu
    :return:
    """

    print(f"Started training at {datetime.now()}")

    results = []
    hyper_param_combinations = itertools.product(models_list, epochs_list, learning_rates)
    for setup in hyper_param_combinations:

        model, epochs, lr = setup

        model_starting_checkpoint = model.state_dict()
        model.load_state_dict(model_starting_checkpoint)

        print(f"Training with setup: [epochs]: {epochs}, [lr]: {lr}, [device]: {device}")
        optimizer = optim.SGD(model.parameters(),
                              lr=lr)
        losses = []
        accs = []
        acc = 0
        loss = 0

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

            acc = calculate_accuracy(test_loader, model, device)
            loss = running_loss / len(train_loader)
            losses.append(loss)
            accs.append(acc)
            print(f"Epoch [{epoch}]: loss: {loss}, train acc {acc}%, Time: {datetime.now()}")

        results.append({
            "accuracy": acc,
            "epochs": epochs,
            "lr": lr,
        })

        torch.save(model.state_dict(), f"./models/emotion_detection_model_epochs_{epochs}_accuracy_{acc}_lr_{lr}.pt")
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds = get_data(0.7, 0.15, 0.15)

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = Model().to(device)
    epochs = [10, 20]
    learning_rates = [0.03, 0.003, 0.3]

    criterion = nn.CrossEntropyLoss()
    results = train([model],
                    epochs,
                    learning_rates=learning_rates,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device)

    print(results)