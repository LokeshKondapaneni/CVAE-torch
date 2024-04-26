# MIT License
# Copyright (c) 2024 Lokesh Kondapaneni

from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Union, Callable, Tuple

CELEB_CLASSES = ['5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes','Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 
                'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 
                'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 
                'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 
                'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling',
                'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
                'Wearing_Necktie', 'Young']

class CelebADataLoader:
    '''
    Responsible for downloading, transforming (preprocess),
    spliting and loading the data

    Returns: Data Loader
    '''
    def __init__(self,
                img_size: int = 65,
                data_path: str = 'data',
                batch_size: int = 64,
                num_workers: int = 0,
                transform: Union[None, Callable] = None) -> None:
        self.img_size = img_size
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform if transform is not None else self.preprocess_transform()

    def preprocess_transform(self) -> Callable:
        """
        Returns a torchvision transform for preprocessing CelebA images.
        
        Returns:
            Callable: A torchvision transform.
        """
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(148),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])

    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates data loaders for the training, validation, and test datasets.
        
        Returns:
            tuple: A tuple containing the data loaders for training, validation, and test datasets.
        """
        self.split_data()
        train_loader = DataLoader(
            self.train_data, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)
        
        valid_loader = DataLoader(
            self.valid_data,
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False)
        
        test_loader = DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False)
        return train_loader, valid_loader, test_loader

    def split_data(self) -> None:
        """
        Splits the CelebA dataset into training, validation, and test sets.
        """
        self.train_data = datasets.CelebA(
            root=self.data_path,
            split='train',
            download=True,
            transform=self.transform)

        self.valid_data = datasets.CelebA(
            root=self.data_path,
            split='valid',
            download=True,
            transform=self.transform)

        self.test_data = datasets.CelebA(
            root=self.data_path,
            split='test',
            download=True,
            transform=self.transform)