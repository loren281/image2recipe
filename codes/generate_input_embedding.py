from load_and_preprocess import Loader
import torch
from PIL import Image
import numpy as np


def get_input_embedding(input_img_filename):
# A function to generate the embedding of the input image
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    model_path = './image2recipe/codes/final_model.pth'  # Path to the pretrained model
    model = Loader().get_pretrained_model(model_path, device)  # Loading the pretrained model using the Loader class

    test_transforms = Loader().data_transforms 

    model.eval() 

    with torch.no_grad():

        img = Image.open(input_img_filename)
        img = test_transforms(img)  # Apply data transformations to the input image
        img = torch.unsqueeze(img, 0)

        embedding = torch.exp(model(img.to(device)))  # Forward pass through the model to find the embedding

        embedding = (embedding - embedding.mean()) / np.std(embedding.numpy())  # Normalize the embedding

        return embedding
