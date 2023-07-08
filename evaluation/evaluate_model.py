import numpy as np
from load_and_preprocess import Loader
import torch
import os
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# Load the pretrained model
model_path = './image2recipe/codes/final_model.pth'
model = Loader().get_pretrained_model(model_path, device)

# Path to the folder containing pairs of imaged associated with the same recipe
root_directory = './image2recipe/data/evaluation'

def predict(img_path):
    # Function to find the embedding of an image

    test_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model.eval()

    with torch.no_grad():
        img = Image.open(img_path)  # Open the image
        img = test_transforms(img)  # Apply transformations to the image
        img = torch.unsqueeze(img, 0)

        embedding = model(img.to(device))  # Get the embedding of the image

    return embedding

files = sorted(os.listdir(root_directory))  # Get a sorted list of files in the root directory
sum_similarity = 0

for i in range(0, len(files), 2):
    file1 = files[i]
    file2 = files[i + 1]

    path1 = os.path.join(root_directory, file1) 
    path2 = os.path.join(root_directory, file2) 

     # Find the embeddings of the two images
    emb1 = predict(path1) 
    emb2 = predict(path2)

    # Normalize the two embeddings
    emb1 = (emb1 - emb1.mean()) / np.std(emb1.numpy())
    emb2 = (emb2 - emb2.mean()) / np.std(emb2.numpy())

     # Calculate the cosine similarity between the embeddings
    cur_similarity = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
   
    similarity_value = cur_similarity[0][0]
    print("current similarity: ", similarity_value)
    sum_similarity += similarity_value

# Calculate the average similarity
avg_similarity = 2 * sum_similarity / len(files)

print("average similarity: ", avg_similarity)
