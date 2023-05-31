import itertools
import os

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
    Trains a set of models with multiple combinations of hyperparameters

    :param model: array of pytorch model classes
    :param epochs: array of epochs numbers
    :param train_loader: dataloader for training data
    :param test_loader: dataloader for test data
    :param val_loader: dataloader for validation data
    :param criterion: loss function
    :param learning_rates: learning rates to be used
    :param device: where to train the model on. cuda or cpu
    :return: dictionary of results
    """

    print(f"Started training at {datetime.now()}")

    results = []
    hyper_param_combinations = itertools.product(models_list, epochs_list, learning_rates)
    for setup in hyper_param_combinations:

        model_class, epochs, lr = setup
        model = model_class()
        model = model.to(device)
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

                # forward pass
                out = model(images)
                loss = criterion(out, targets)

                # backward pass/ calculating gradients
                loss.backward()
                running_loss += loss.item()

                # adjusting weights
                optimizer.step()

            acc = calculate_accuracy(test_loader, model, device)
            loss = running_loss / len(train_loader)
            losses.append(loss)
            accs.append(acc)
            print(f"Epoch [{epoch}]: loss: {loss}, test acc {acc}%, Time: {datetime.now()}")

        results.append({
            "accuracy": acc,
            "epochs": epochs,
            "lr": lr,
        })

        torch.save(model.state_dict(), f"./models/emotion_detection_model_epochs_{epochs}_accuracy_{acc}_lr_{lr}.pt")
    return results


def validate_models(dl, device):

    """
    validates the models in the ./models folder with the validation set
    :param dl: data loader for validation set
    :param device: device the models should be validated on
    :return:
    """
    models = []
    results = []

    for model_name in os.listdir("./models"):
        mc = Model().to(device)
        mc.load_state_dict(torch.load("./models/" + model_name))
        mc.eval()
        models.append(mc)
        results.append({"name": model_name,
                        "validation_accuracy": calculate_accuracy(dl, mc, device)
                        })
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds = get_data(0.7, 0.15, 0.15)

    train_loader = DataLoader(train_ds, batch_size=32)
    test_loader = DataLoader(train_ds, batch_size=32)
    val_loader = DataLoader(val_ds, batch_size=32)

    epochs = [10, 20]
    learning_rates = [0.03, 0.003, 0.3]

    criterion = nn.CrossEntropyLoss()


    validation_results = validate_models(val_loader, device)
    print(validation_results)
