from .datasets import MyDataset
from torch.utils.data import DataLoader


def get_loaders(config, learning_phase):
    if learning_phase:
        train_dataset = MyDataset(
            subdir="train",
            list_custom_classes_training_phase=config.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=None,
            root_path_images=config.root_path_images,
            to_transform=config.to_transform,
            imgs_format=config.imgs_format,
            training_phase=True,
        )

        validation_dataset = MyDataset(
            subdir="test",
            list_custom_classes_training_phase=config.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=None,
            root_path_images=config.root_path_images,
            to_transform=config.to_transform,
            imgs_format=config.imgs_format,
            training_phase=True,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

        return train_dataset, validation_dataset, train_loader, validation_loader

    else:
        eval_dataset = MyDataset(
            subdir="test",
            list_custom_classes_training_phase=config.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=config.list_custom_classes_eval_phase,
            root_path_images=config.root_path_images,
            to_transform=config.to_transform,
            imgs_format=config.imgs_format,
            training_phase=False,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )
        return eval_loader
