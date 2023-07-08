import numpy as np
from load_and_preprocess import Loader
import torch
import os


model_path = './image2recipe/codes/final_model.pth'  # Path to the pretrained model
data_path = './image2recipe/data/images'  # Path to the directory with the dataset's images
batch_size = 64

embeddings_dir = './image2recipe/embeddings'  # Directory to save the embeddings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

dataloader = Loader().get_dataloader(data_path, batch_size)  # Create the dataloader

model = Loader().get_pretrained_model(model_path, device)

num_batches = len(dataloader)  # Total number of batches in the data loader
num_batches_to_save = 100  # Number of batches after which to save the embeddings
file_num = 0  # This counter will be used in the embeddings file names

embedding_dict = dict()  # Dictionary to store filename-embedding pairs

with torch.no_grad():  
    for i, data in enumerate(dataloader):

        images = data['data']  # Extract the images from the batch
        batch_filenames = data['filename']  # Extract the filenames from the batch

        images = images.to(device)

        embeddings = model(images)

        # Store the filename-embedding pair in the dictionary
        for j in range(embeddings.shape[0]):
            filename = batch_filenames[j]
            emb = embeddings[j]
            embedding_dict[filename] = emb  

        print("batch: " + str(i + 1) + "/" + str(num_batches))

        # Save embeddings in a file once every 100 batches
        if (i + 1) % num_batches_to_save == 0:
            print('Saving embedding: ' + str(file_num))
            file_path = os.path.join(embeddings_dir, 'emb_'+ str(file_num))
            np.save(file_path, embedding_dict)
            file_num += 1 
            embedding_dict = dict()

# Check if there are any remaining embeddings to be saved (if number of batches didn't divide by 100 exactly)
if len(embedding_dict) > 0:  
    print('Saving embedding: ' + str(file_num))
    file_path = os.path.join(embeddings_dir, 'emb_'+ str(file_num))
    np.save(file_path, embedding_dict)
