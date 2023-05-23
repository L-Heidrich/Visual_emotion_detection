from model_class import Model
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from data import get_data
from torch.utils.data import DataLoader


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


def transform_image(img):
    """
    :param img: PIL image
    :return:
    """
    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])

    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    return img


if __name__ == "__main__":
    print("loading model")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model().to(device)
    model.load_state_dict(torch.load("./models/emotion_detection_model.pt", map_location=torch.device(device)))
    model.eval()

    print("testing accuracy")

    train_ds, test_ds = get_data()
    test_loader = DataLoader(train_ds, batch_size=32)
    #print(f"Accuracy on test data: {calculate_accuracy(test_loader, model)}%")

    print("Test on random image")
    img = Image.open("./test_images/happy-woman-2.jpg")
    img = transform_image(img)

    #plt.imshow(img.permute(1, 2, 0))
    #plt.show()

    outputs = model(img)
    _, preds = torch.max(outputs, dim=1)

    print(outputs, model.classes[preds.item()])
