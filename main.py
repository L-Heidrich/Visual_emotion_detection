import gc
import itertools
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data import get_data
import matplotlib.pyplot as plt
from model_class import Model, Model_small, Model_big
from datetime import datetime


def calculate_accuracy(dataloader, model, device):
    """
    :param dataloader: dataloader you want to measure the accuracy on
    :param model: Your model
    :return: accuracy as an int
    """

    model.eval()
    correct_images = 0
    incorrect_images = 0

    total_images = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["data"].to(device)
            targets = batch["label"].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(targets, outputs)
            total_images += targets.size(0)
            correct_images += (predicted == targets).sum().item()
            incorrect_images += (predicted != targets).sum().item()

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
        print(f"Training with setup: [epochs]: {epochs}, [lr]: {lr}, [device]: {device}, [model]: {model.name}")
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=0.9)
        val_losses = []
        val_accs = []
        best_val_acc = 0.0
        epochs_since_improvement = 0

        for epoch in range(epochs):
            running_loss = 0.0
            model = model.train()

            for batch in train_loader:
                optimizer.zero_grad()

                images = batch["data"].to(device)
                targets = batch["label"].to(device)

                # forward pass
                out = model(images)
                loss = criterion(out, targets)

                # backward pass/ calculating gradients
                loss.backward()
                running_loss += loss.item()

                # adjusting weights
                optimizer.step()

            val_acc = calculate_accuracy(val_loader, model, device)
            val_loss = running_loss / len(train_loader)
            val_accs.append(val_acc)
            val_losses.append(val_loss)

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epochs_since_improvement >= 20:
                print("Early stopping triggered. No improvement in validation accuracy for 20 epochs.")
                break

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}]: loss: {val_loss}, validation acc {val_acc}%, Time: {datetime.now()}")

        test_acc = calculate_accuracy(test_loader, model, device)
        print(f"Test accuracy: {test_acc}%")

        results.append({
            "accuracy": test_acc,
            "epochs": epoch,
            "lr": lr,
            "model": {model.name}
        })

        torch.save(model.state_dict(),
                   f"./models/emotion_detection_model_epochs_{epochs}_accuracy_{test_acc}_lr_{lr}.pt")
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
    train_ds, val_ds, test_ds = get_data(0.8, 0.1, 0.1)

    train_loader = DataLoader(train_ds, batch_size=16)
    test_loader = DataLoader(train_ds, batch_size=16)
    val_loader = DataLoader(val_ds, batch_size=16)

    epochs = [200]
    learning_rates = [0.003]

    gc.collect()
    torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    models_list = [Model_small]

    results = train(models_list=models_list,
                    epochs_list=epochs,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    criterion=criterion,
                    learning_rates=learning_rates,
                    device=device)
    #validation_results = validate_models(val_loader, device)
    #print(validation_results)
