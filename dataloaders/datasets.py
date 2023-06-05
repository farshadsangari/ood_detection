import os
import numpy as np
import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import random
from PIL import Image
import torch


class MyDataset(Dataset):
    def __init__(
        self,
        subdir,
        list_custom_classes_training_phase,
        list_custom_classes_eval_phase,
        root_path_images,
        to_transform,
        imgs_format,
        eval_phase,
    ):
        super(MyDataset, self).__init__()
        self.root_path_images = root_path_images
        self.subdir = subdir
        self.to_transform = to_transform
        self.imgs_format = imgs_format
        self.class_to_int = {
            list_custom_classes_training_phase[i]: i
            for i in range(len(list_custom_classes_training_phase))
        }
        if subdir == "train":
            self.transformations = self.transformations_train
        elif subdir == "test":
            self.transformations = self.transformations_test
        else:
            raise Exception("Unknown subdir")

        if eval_phase:
            self.path_images = self.get_custom_classes_path(
                custom_classes=list_custom_classes_eval_phase
            )
            for class_ in list_custom_classes_eval_phase:
                if class_ not in self.class_to_int.keys():
                    self.class_to_int[class_] = max(self.class_to_int.values()) + 1
        else:
            self.path_images = self.get_custom_classes_path(
                custom_classes=list_custom_classes_training_phase
            )

    def transformations_train(self, to_tensor, last_idx_before_tensor=1):
        list_image_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
        ]

        if to_tensor:
            return transforms.Compose(list_image_transforms)

        else:
            return transforms.Compose(
                list_image_transforms[: last_idx_before_tensor + 1]
            )

    def transformations_test(self, to_tensor, last_idx_before_tensor=-1):
        list_image_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.49139968, 0.48215827, 0.44653124],
                std=[0.24703233, 0.24348505, 0.26158768],
            ),
        ]

        if to_tensor:
            return transforms.Compose(list_image_transforms)

        else:
            return transforms.Compose(
                list_image_transforms[: last_idx_before_tensor + 1]
            )

    def __getitem__(self, index):
        # index ---> this is in range 0-len(train/test dataset) and should be converted into real_index(because of label2factor index function)
        _, image_tensor, label = self.read_image(index)

        return image_tensor, self.class_to_int[label]

    def get_custom_classes_path(self, custom_classes):
        paths = []
        available_classes_in_dir = os.listdir(
            os.path.join(self.root_path_images, self.subdir)
        )
        # if custom_classes not in available_classes_in_dir:
        #     raise FileNotFoundError(
        #         "Some of custom clsses dont exists in your directory"
        #     )
        for _class in custom_classes:
            paths += glob.glob(
                os.path.join(self.root_path_images, self.subdir, _class)
                + f"/*{self.imgs_format}"
            )
        paths.sort()
        return paths

    def read_image(self, index):
        image_path = self.path_images[index]
        image = Image.open(image_path)
        image, image_tensor = self.transformation_handler(image)
        label = image_path.split("/")[-2]
        return image, image_tensor, label

    def transformation_handler(self, image):
        if self.to_transform:
            transformed_image = self.transformations(to_tensor=False)(image)
            transformed_tensor = self.transformations(to_tensor=True)(image)
            return transformed_image, transformed_tensor
        else:
            return image, transforms.ToTensor()(image)

    def __len__(self):
        return len(self.path_images)

    def sample(self, num):
        rand_indexes = random.sample(range(len(self.path_images)), k=num)
        imgs_list, labels_lst = [], []
        for index in rand_indexes:
            _, image_tensor, label = self.read_image(index)
            imgs_list.append(image_tensor)
            labels_lst.append(label)
        images_batch = torch.stack(imgs_list, dim=0)
        return images_batch, labels_lst

    def get_custom_classes_data(self, custom_classes):
        for class_ in custom_classes:
            if class_ not in self.all_classes_to_int.keys():
                self.all_classes_to_int[class_] = max(self.class_to_int.values()) + 1
        img_paths = self.get_custom_classes_path(custom_classes=custom_classes)
        imgs_list, labels_lst = [], []
        for path in img_paths:
            image = Image.open(path)
            image, image_tensor = self.transformation_handler(image)
            label = path.split("/")[-2]
            label = self.class_to_int[label]
            imgs_list.append(image_tensor)
            labels_lst.append(label)
        imgs_batchwise_form = torch.stack(imgs_list, dim=0)
        labels_tensor = torch.tensor(labels_lst)

        return imgs_batchwise_form, labels_tensor
