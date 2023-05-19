from datasets import load_dataset, Dataset
from torchvision import transforms


def get_data():
    dataset = load_dataset("FastJobs/Visual_Emotional_Analysis", split="train")

    dataset = dataset.shuffle(seed=432)
    train_ds = dataset[:640]
    test_ds = dataset[640:]

    label_count = [0, 0, 0, 0, 0, 0, 0, 0]

    for label in train_ds["label"]:
        label_count[label] += 1

    print(label_count)
    my_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
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

    return train_ds, test_ds


get_data()
