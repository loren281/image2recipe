import torch
from torchvision import datasets, transforms, models
import os
from torch import nn


class MyDataset:
# Creating a costume dataset, with the "data" being the images, and the "labels" being the recipe id (which is the image name)
    def __init__(self, root, transforms):
        self.root = root
        self.dataset = datasets.ImageFolder(root=self.root, transform=transforms)

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        return {
            "data": data,
            "filename": os.path.basename(self.dataset.imgs[index][0])  # The filename of the data is actually the recipe id
        }

    def __len__(self):
        return len(self.dataset)  # Returning the length of the dataset


class Loader:
    def __init__(self):
        self.data_transforms = transforms.Compose([transforms.Resize((256,256)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                        [0.229, 0.224, 0.225])])


    def get_dataloader(self, data_path, batch_size, shuffle=False):
    # Create a data loader of the costume dataset
        data_transforms = self.data_transforms 

        dataset = MyDataset(data_path, data_transforms)  # Create the custom dataset

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                  generator=torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu"))

        return dataloader
    
    def get_pretrained_model(self, model_path, device):
    # Return a model with the architecture modifications used in the transfer learning
        model = models.densenet121(weights='IMAGENET1K_V1')  # Load the pre-trained DenseNet-121 model

        model.classifier = nn.Sequential(nn.Linear(1024,101))
        model.load_state_dict(torch.load(model_path, map_location=device))  # Loading the pre-trained model weights

        return model
