import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

class RecipeInfo:
    def __init__(self):
        self.data = pd.read_json('C:/Users/loren/Desktop/dl_project/image2recipe/data/layer1.json') # This file is part of the recipe1M dataset. Contains the ingredients, recipe ID, image url...

    def get_recipe_info(self, recipe_id):
    # Finds the recipe of the given image
        recipe_info = self.data.loc[self.data['id'] == recipe_id]
        return recipe_info
    
    def display_recipe(self, recipe_info):
    # Displays the recipe's ingredients and instructions
        print('\033[37m' + '\033[1m' + '\033[4m' + recipe_info['title'].values[0] + '\033[0m')
        print('\033[37m' + "Ingredients:" + '\033[0m')
        ingredients = recipe_info['ingredients'].values[0]
        for ingredient in ingredients:
            print(ingredient['text'])
        print('\n')
        print('\033[37m' + "Instructions:" + '\033[0m')
        instructions = recipe_info['instructions'].values[0]
        for inst in instructions:
            print(inst['text'])
    
    def display_food_images(self, dataset_dir, best_files):
    # Display the chosen food images
        for img in best_files:
            img_path = os.path.join(dataset_dir, img)
            img = cv2.imread(img_path)
            plt.imshow(img[...,::-1])
            plt.xticks([])
            plt.yticks([])
            plt.show()
    
    def print_best_recipes(self, best_files):
    # Print the chosen recipes
        for recipe_num, file in enumerate(best_files):
            print('\033[1m' + '\033[94m' + "Recipe " + str(recipe_num + 1) + '\033[0m')
            file_name = file[:-4]
            recipe_info = self.get_recipe_info(str(file_name))
            self.display_recipe(recipe_info)
            print()

