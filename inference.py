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


print("loading model")
model = Model().to("cuda")
model.load_state_dict(torch.load("./models/emotion_detection_model.pt"))
model.eval()

print("testing accuracy")

train_ds, test_ds = get_data()
test_loader = DataLoader(train_ds, batch_size=32)
print(f"Accuracy on test data: {calculate_accuracy(test_loader, model)}%")

print("Test on random image")
img = Image.open("./test_images/happy-woman-2.jpg")
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

img = transform(img)
plt.imshow(img.permute(1, 2, 0))
plt.show()

img = torch.unsqueeze(img, 0)

img = img.to('cuda')

outputs = model(img)
_, preds = torch.max(outputs, dim=1)

print(outputs, model.classes[preds.item()])
