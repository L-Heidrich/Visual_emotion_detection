from datasets import load_dataset, Dataset
from torchvision import transforms


def get_data(train_percentile, val_percentile, test_percentile):
    dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")

    dataset = dataset.shuffle(seed=432)

    if (train_percentile + val_percentile + test_percentile) != 1:
        print("Split distributions must sum up to one")
    else:
        train_idx = int(train_percentile * len(dataset))
        val_idx = int(val_percentile * len(dataset))
        test_idx = int(test_percentile * len(dataset))

        train_ds = dataset[:train_idx]
        test_ds = dataset[train_idx:train_idx + test_idx]
        val_ds = dataset[train_idx + test_idx:]

        my_transforms = transforms.Compose(
            [
                # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.ToTensor()]
        )

        my_test_transforms = transforms.Compose(
            [
                transforms.ToTensor()]
        )

        def apply_transforms(images):
            images["image"] = [my_transforms(image) for image in images["image"]]
            return images

        def apply_test_transforms(images):
            images["image"] = [my_test_transforms(image) for image in images["image"]]
            return images

        train_ds = Dataset.from_dict(train_ds).with_format("torch").with_transform(apply_transforms)
        test_ds = Dataset.from_dict(test_ds).with_format("torch").with_transform(apply_test_transforms)
        val_ds = Dataset.from_dict(val_ds).with_format("torch").with_transform(apply_test_transforms)

        return train_ds, val_ds, test_ds


if __name__ == "__main__":

    train_ds, val_ds, test_ds = get_data(0.8, 0.1, 0.1)

    labels = [0, 0, 0, 0, 0, 0, 0, 0]
    for entry in train_ds:
        labels[entry["label"]] += 1

    print(labels)
