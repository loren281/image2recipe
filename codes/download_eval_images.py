import json
import requests
import os
from PIL import Image

# Function to download an image from a given URL
def download_image(url, file_path, image_size):
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    try:
        image = Image.open(file_path)
        image = image.resize(image_size)
        image = image.convert('RGB')
        image.save(file_path)
        return 1

    except:
        print("corrupted image: {}\n".format(file_path))
        return 0
    
image_size = (256, 256)
root = './image2recipe/data/evaluation'

# Read the json file which contains the url and recipe id
with open('./image2recipe/data/layer2.json') as file:
    data = json.load(file)

downloaded_images = 0

for line in data:
    if 'images' in line and len(line['images']) >= 2:
        urls = [image['url'] for image in line['images'][:2]]
        filename1 = f"{line['id']}_1.jpg"
        filename2 = f"{line['id']}_2.jpg"
        path1 = os.path.join(root, filename1)
        path2 = os.path.join(root, filename2)

        # Download pairs of images with the same recipe
        downloaded_images += download_image(urls[0], path1, image_size)
        downloaded_images += download_image(urls[1], path2, image_size)

        print("downloaded images: ", downloaded_images)

        if downloaded_images >= 2000:
            break
