# image2recipe

An implementation of image2recipe - a deep learning based system that outputs the recipe of a given food image.
We use DenseNet121 pretrained on ImageNet, and through the technique of transfer learning we further train the model on the Food-101 dataset, to enhance its ability to understand food-related features.
This model is used for generating the embeddings for the dataset's images.
Given an input image, we search for the closest embeddings to its embedding, and output the recipes of those top images. 

## Repository Organization

|File name         | Content |
|----------------------|------|
|`/codes/image2recipe.ipynb`| main wrapper code which the user can use to input their image and receive its recipe|
|`/codes/transer_learning_densenet121.ipynb`| code that performs transfer learning on DenseNet121 model|
|`/codes/generate_train_embeddings.py`| code that generates the trainset embeddings using the pretrained model|
|`/codes/generate_input_embedding.py`| code that generates the input image embedding using the pretrained model|
|`/codes/load_and_preprocess.py`| class for creating data loaders of a costume dataset, and for loading the pretrained model|
|`/codes/recipe_info.py`| class for locating the chosen recipes, printing them and displaying the top images|
|`/codes/embedding_loader.py`| class for finding the closest embeddings to the input image's embedding|
|`/codes/nearest_neighbors.py`| code that calculates the similarity between given embeddings|
|`/codes/download_eval_images.py`| code for downloading pairs of images associated with the same recipe|
|`/codes/evaluate_model.py`| code for evaluating the model|
|`/model/final_model.pth`| our final pretrained model (DenseNet121 after transfer learning)|

## Dataset
We are using the [Recipe1M](http://im2recipe.csail.mit.edu/) dataset which is not included in this repository.

## Running Instructions

Make sure to modify all paths to match your work area.

1. **Train the model**

- Run the code: ```transer_learning_densenet121.ipynb```

This code uses the pretrained DenseNet121, while replacing its last layer with a fully-connected layer of size 101.
You can modify the hyperparameters and the architecture as you wish.

- Make sure to: modify the path in which the final model will be saved at.
- After this step you will have a file called "final_model.pth".
- Alternatively, you can use our pretrained model which is located at: ```model/final_model.pth```

2. **Generate dataset embeddings**

- Run the code: ```generate_train_embeddings.py```

This code generates the embeddings for the entire dataset.

- Make sure to: create a directory to store the embeddings and modify paths.
- After this step you will have the dataset's embeddings.

3. **Generate and display top recipes and images**

- Run the code: ```image2recipe.ipynb```

- Make sure to: modify paths including the desired input image path, and choose number of recipes to display.
- After this step the desired recipes will be displayed. 

