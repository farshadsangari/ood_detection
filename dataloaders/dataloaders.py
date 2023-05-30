from .datasets import MyDataset
from torch.utils.data import DataLoader


def get_loaders(configs, eval_phase):
    if not eval_phase:
        train_dataset = MyDataset(
            subdir="train",
            list_custom_classes_training_phase=configs.dataloader.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=None,
            root_path_images=configs.dataloader.root_path_images,
            to_transform=configs.dataloader.to_transform,
            imgs_format=configs.dataloader.imgs_format,
            eval_phase=False,
        )

        validation_dataset = MyDataset(
            subdir="test",
            list_custom_classes_training_phase=configs.dataloader.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=None,
            root_path_images=configs.dataloader.root_path_images,
            to_transform=configs.dataloader.to_transform,
            imgs_format=configs.dataloader.imgs_format,
            eval_phase=False,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=configs.dataloader.batch_size,
            shuffle=True,
        )
        validation_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=configs.dataloader.batch_size,
            shuffle=True,
        )

        return train_dataset, validation_dataset, train_loader, validation_loader

    else:
        eval_dataset = MyDataset(
            subdir="test",
            list_custom_classes_training_phase=configs.dataloader.list_custom_classes_training_phase,
            list_custom_classes_eval_phase=configs.eval.list_custom_classes_eval_phase,
            root_path_images=configs.eval.root_path_images,
            to_transform=configs.eval.to_transform,
            imgs_format=configs.eval.imgs_format,
            eval_phase=True,
        )
        eval_loader = DataLoader(
            dataset=eval_dataset,
            batch_size=configs.dataloader.batch_size,
            shuffle=True,
        )
        return eval_loader
