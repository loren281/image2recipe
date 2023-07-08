# Original code is from: https://github.com/slu1212/cs230, with our modifications

import numpy as np
import os
from nearest_neighbors import nearest_neighbors, display_progress, update_best


class EmbeddingLoader:
    
    def __init__(self, emb_dir):
        # Initialize the EmbeddingLoader with the directory containing the embeddings
        self.emb_dir = emb_dir
        self.num_enbeddings = self.get_num_embeddings()  # Get the number of embedding files in the directory
        self.cur_file = 0  # Initialize the current file index to 0
    
    def get_num_embeddings(self):
        # Count the number of embedding files in the directory
        file_count = 0
        for file in os.listdir(self.emb_dir):
            if(file.endswith('.npy')):
                file_count += 1
        return file_count
        

    def reset_iter(self):
        # Reset the current file index to 0
        self.cur_file = 0
        
    
    def load_next_embedding(self):
        # Load the next embedding file in the directory
        if self.cur_file == self.num_enbeddings:
            # If all files have been loaded, reset the current file index and return None
            self.cur_file = 0
            return None

        filename = f'emb_{self.cur_file}.npy'
        file_path = os.path.join(self.emb_dir, filename)
        emb_dict = np.load(file_path, allow_pickle=True).item()  # Load the embedding dictionary from the file

        emb_matrix = np.vstack(list(emb_dict.values()))  # Stack the embedding vectors into a matrix

        index_to_filename = {idx: img_file for idx, img_file in enumerate(emb_dict.keys())}
        # Create a mapping from index to filename for the loaded embeddings

        self.cur_file += 1
        return emb_matrix, index_to_filename

    
    def load_enbeddings_from_file(self, filename):
        # Load embeddings from a specific file
        file = np.load(self.emb_dir + '/' + filename, allow_pickle=True)
        emb_dict = file[()]
        length = emb_dict[list(emb_dict.keys())[0]].shape[0]
        print("length: ", length)
        emb_matrix = np.zeros((len(emb_dict), length))
        index_to_filename = dict()
        idx = 0
        for img_file, enbedding in emb_dict.items():
            index_to_filename[idx] = img_file
            emb_matrix[idx] = enbedding
            idx += 1
        return emb_matrix, index_to_filename
    

    def find_best_embedding(self, num_recipes, input_embedding, metric):
        # Find the best matching embeddings based on the input embedding and similarity metric
        self.reset_iter()  # Reset the current file index
        best_files = [None] * num_recipes  # Initialize a list to store the best matching file names
        best_sims = [None] * num_recipes  # Initialize a list to store the best matching similarities
        i = 0

        while True:
            display_progress(i, self.num_enbeddings)  # Display progress during the iteration
            i += 1

            embedding = self.load_next_embedding()  # Load the next embedding
            if embedding is None:
                break  # If all files have been processed, exit the loop

            emb_matrix, index_to_filename = embedding

            row_means = np.mean(emb_matrix, axis=1)  
            row_stds = np.std(emb_matrix, axis=1) 

            # Normalize the embeddings matrix
            emb_matrix = (emb_matrix - row_means[:, np.newaxis]) / row_stds[:, np.newaxis]
            
            # Find the nearest neighbors to the input embedding using the specified metric
            idx, similarity = nearest_neighbors(emb_matrix, np.squeeze(input_embedding), metric)
           
            # Update the best matching file names and similarities based on the current embedding
            best_files, best_sims = update_best(best_files, best_sims, similarity, index_to_filename[idx], metric)
            
        print('\n')

        for i in range(num_recipes):
            best_sims[i] = (best_sims[i]+1)/2  # Normalize results between [0,1]
            print(f'img {i + 1}: {best_files[i]}')
            print(f'similarity: {best_sims[i]}')
        return best_files
