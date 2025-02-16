# =============================================================================
#                             Import Standard Packages
# =============================================================================
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
from PIL import Image
import logging
import os

import matplotlib.pyplot as plt


# =============================================================================
#                             Configuration
# =============================================================================
config = {
    'DATA_DIR': './Data/raw',                          # Path to folder with original images
    'SAVE_DIR': './Data/processed/random_samples',     # Destination folder for random sample images
    'DEBUG': False,                                    # Flag to enable/disable debug output
}


# =============================================================================
#                           Logger Configuration
# =============================================================================
# Set up root logger for all modules (including third-party libraries)
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
root_handler = logging.StreamHandler()
root_handler.setFormatter(logging.Formatter('%(asctime)s - ROOT - %(levelname)s - %(message)s'))
root_logger.addHandler(root_handler)


# =============================================================================
#                   Module-specific Logging Configuration
# =============================================================================
# Set up a dedicated logger for this module
my_module_logger = logging.getLogger(__name__)
if config['DEBUG'] == True:
    my_module_logger.setLevel(logging.DEBUG)
else:
    my_module_logger.setLevel(logging.INFO)
my_module_logger.propagate = False                      # Prevent duplicate log entries

# Create and configure a stream handler for this module's logger
my_module_handler = logging.StreamHandler()
my_module_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
my_module_logger.addHandler(my_module_handler)


# =============================================================================
#                          Class: RandomeImagesData
# =============================================================================
class RandomeImagesData:
    '''
    Class to load and handle a random subset of images from a directory.
    
    Utilizes torchvision's ImageFolder to load images and creates a DataLoader
    that provides a random subset of the available images.
    '''

    def __init__(self, data_dir: str = config['DATA_DIR'], num_samples: int = 10):
        '''
        Initialize the RandomeImagesData object.

        Args:
            data_dir (str): Directory containing the image data.
            num_samples (int): Number of random samples to select.
        '''
        self.data_dir = data_dir

        # Load the image dataset from the specified directory and apply a basic tensor transform      
        self.random_dataset = datasets.ImageFolder(root = data_dir, transform = transforms.ToTensor())
        # Generate a DataLoader with a randomly selected subset of images
        self.random_loader = self.get_dataloaders(self.random_dataset, num_samples)
        # Save the list of class names from the dataset
        self.classes = self.random_dataset.classes


    def get_dataloaders(self, random_dataset, num_samples):
        '''
        Create a DataLoader for a random subset of the given dataset.

        Args:
            random_dataset: The full image dataset.
            num_samples (int): The number of random images to select.

        Returns:
            DataLoader: DataLoader for the random subset.
        '''

        # Create a list of indices corresponding to each image and shuffle indices to ensure randomness
        dataset_size = len(random_dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)

        # Select the first 'num_samples' indices after shuffling.
        random_indices = indices[:num_samples]

        my_module_logger.debug(f'Random dataset indices: {min(random_indices):d}-{max(random_indices):d}, ' 
                               f'Number of instances: {len(random_indices)}')           

        # Create a DataLoader using a Subset of the dataset defined by the random indices.
        randome_loader  = DataLoader(Subset(random_dataset, random_indices),
                                batch_size = 1, shuffle = True, 
                                )
        return randome_loader


# =============================================================================
#                       Function to save images
# =============================================================================
def save_random_images(data, save_dir: str = config['SAVE_DIR']):
    '''
    Save images from the random DataLoader to the specified directory.

    Args:
        data (RandomeImagesData): Data object containing the random DataLoader.
        save_dir (str): Directory where images will be saved. Default is a preconfigured value of 'SAVE_DIR'.

    Returns:
        list: Filenames of the saved images.
    '''    
    file_names = []
    # Iterate over the DataLoader batches (each containing one image and its label)    
    for batch_idx, (image, label) in enumerate(data.random_loader):
        # Convert tensor image to a PIL image for saving and display
        to_PIL = transforms.ToPILImage()
        # Retrieve the class name for the current image and generate a unique filename
        garbage_class = data.classes[label]
        filename = f'{garbage_class}_{batch_idx}.jpg'

        # Save the image to the target directory
        to_PIL(image.squeeze(0)).save(os.path.join(save_dir, filename))
        file_names.append(filename)

        # If running as a standalone script, display the image using matplotlib.
        if __name__ == '__main__':
            plt.imshow(to_PIL(image.squeeze(0)))
            plt.title(garbage_class + filename)
            plt.show()

    my_module_logger.debug(f"Saved images to {save_dir}")
    return file_names


# =============================================================================
#                       Generate and save randome images 
# =============================================================================
def select_random_images(source_dir: str, target_dir: str, num: int = 10):
    data = RandomeImagesData(source_dir, num_samples = num)
    return save_random_images(data, target_dir)


# =============================================================================
#     When this script is executed directly, select and save random images
# =============================================================================
if __name__ == '__main__':
    files = select_random_images(config['DATA_DIR'], config['SAVE_DIR'])
    print(files)



