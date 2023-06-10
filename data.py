import torch
from datasets import load_dataset, Dataset
from torchvision import transforms
import torch.nn.functional as F

def get_data(train_percentile = None, val_percentile = None, test_percentile= None):
    dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")

    dataset = dataset.shuffle()
    
    if (train_percentile + val_percentile + test_percentile) != 1:
        print("Split distributions must sum up to one")
    else:
        train_idx = int(train_percentile * len(dataset))
        val_idx = int(val_percentile * len(dataset))
        test_idx = int(test_percentile * len(dataset))

        train_ds = dataset[:train_idx]
        test_ds = dataset[train_idx:train_idx + test_idx]
        val_ds = dataset[train_idx + test_idx:]

    """
    train_ds = dataset["train"]
    test_ds = dataset["test"]
    val_ds = dataset["validation"]
    """

    my_transforms = transforms.Compose(
        [
            transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 180)),
            #transforms.RandomGrayscale(p=0.5),
            transforms.ToTensor()]
    )

    my_test_transforms = transforms.Compose(
        [
            transforms.ToTensor()]
    )

    def apply_transforms(images):
        images["data"] = [my_transforms(image) for image in images["data"]]
        images["label"] = [torch.Tensor(label) for label in images["label"]]
        return images

    def apply_test_transforms(images):
        images["data"] = [my_test_transforms(image) for image in images["data"]]
        images["label"] = [torch.Tensor(label) for label in images["label"]]

        return images

    train_ds = Dataset.from_dict({"data": train_ds["image"], "label": F.one_hot(torch.Tensor(train_ds["label"]).to(torch.int64), 8)}).with_format("torch").with_transform(apply_transforms)
    test_ds = Dataset.from_dict({"data": test_ds["image"], "label": F.one_hot(torch.Tensor(test_ds["label"]).to(torch.int64), 8)}).with_format("torch").with_transform(apply_test_transforms)
    val_ds = Dataset.from_dict({"data": val_ds["image"], "label": F.one_hot(torch.Tensor(val_ds["label"]).to(torch.int64), 8)}).with_format("torch").with_transform(apply_test_transforms)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_data(0.8, 0.1, 0.1)

    for entry in val_ds:
        print(entry)
        break
