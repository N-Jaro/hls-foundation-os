import os
import numpy as np
import json
from PIL import Image
from patchify import patchify
import tensorflow as tf

# # Example of how to use the class
# data_loader = DataLoader('/path/to/AR_Maumee.tif', '/path/to/AR_Maumee.json', patch_size=(256, 256, 3), overlap=30)
# processed_data = data_loader.get_processed_data()
# reconstructed_image = data_loader.reconstruct_data(processed_data['map_patches'])

class DataLoader:
    """
    DataLoader class to load and process TIFF images and corresponding JSON files.

    Attributes:
    tiff_path (str): Path to the input TIFF image file.
    json_path (str): Path to the input JSON file.
    patch_size (tuple of int): Size of the patches to extract from the TIFF image.
    overlap (int): Overlapping pixels between patches.
    map_img (numpy.ndarray): Loaded and normalized map image.
    orig_size (tuple of int): Original size of the map image.
    json_data (dict): Loaded JSON data.
    processed_data (dict): Processed data including map patches and legends.

    Methods:
    load_and_process(): Loads and processes the TIFF image and JSON data.
    load_tiff(): Loads and normalizes the TIFF image.
    load_json(): Loads JSON data.
    process_legends(label_suffix, resize_to=(256, 256)): Processes legends based on label suffix.
    process_data(): Extracts and processes data from the loaded TIFF and JSON.
    get_processed_data(): Returns the processed data.
    reconstruct_data(self, patches): unpatchify the prediction data. 
    """
    def __init__(self, tiff_path, json_path, patch_size=(224, 224, 3), overlap=30):
        """
        Initializes DataLoader with specified file paths, patch size, and overlap.

        Parameters:
        tiff_path (str): Path to the input TIFF image file.
        json_path (str): Path to the input JSON file.
        patch_size (tuple of int, optional): Size of patches to extract from image. Default is (256, 256, 3).
        overlap (int, optional): Number of overlapping pixels between patches. Default is 30.
        """
        self.tiff_path = tiff_path
        self.json_path = json_path
        self.patch_size = patch_size
        self.overlap = overlap
        self.map_img = None
        self.orig_size = None
        self.json_data = None
        self.processed_data = None
        self.load_and_process()

    def load_and_process(self):
        """Loads and processes the TIFF image and JSON data."""
        self.load_tiff()
        self.load_json()
        self.process_data()

    def load_tiff(self):
        """Loads and normalizes the TIFF image."""
        print(f"Loading and normalizing map image: {self.tiff_path}")
        self.map_img = Image.open(self.tiff_path)
        self.orig_size = self.map_img.size  
        self.map_img = np.array(self.map_img) #/ 255.0

    def load_json(self):
        """Loads JSON data."""
        with open(self.json_path, 'r') as json_file:
            self.json_data = json.load(json_file)

    def process_lgeends(self, label_suffix):
        """
        Processes legends based on the label suffix.

        Parameters:
        label_suffix (str): Suffix in the label to identify the type of legend.
        resize_to (tuple of int, optional): Size to resize the legends to. Default is (256, 256).

        Returns:
        list of tuples: List of processed legends and their corresponding labels.
        """
        legends = [shape for shape in self.json_data['shapes'] if label_suffix in shape['label']]
        processed_legends = []
        for legend in legends:
            points = np.array(legend['points'])
            top_left = points.min(axis=0)
            bottom_right = points.max(axis=0)
            legend_img = self.map_img[int(top_left[1]):int(bottom_right[1]), int(top_left[0]):int(bottom_right[0]), :]
            legend_img = tf.image.resize(legend_img, self.patch_size[:3])
            processed_legends.append((legend_img.numpy(), legend['label']))
        return processed_legends

    def process_data(self):
        """Extracts and processes data from the loaded TIFF and JSON."""
        step_size = self.patch_size[0] - self.overlap
        
        pad_x = (step_size - (self.map_img.shape[1] % step_size)) % step_size
        pad_y = (step_size - (self.map_img.shape[0] % step_size)) % step_size
        self.map_img = np.pad(self.map_img, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant')

        print(f"Patchifying map image with overlap...")
        map_patches = patchify(self.map_img, self.patch_size, step=step_size)

        poly_legends = self.process_legends('_poly')
        pt_legends = self.process_legends('_pt')
        line_legends = self.process_legends('_line')

        self.processed_data = {
            "map_patches": map_patches,
            "poly_legends": poly_legends,
            "pt_legends": pt_legends,
            "line_legends": line_legends,
            "original_size": self.orig_size,
        }

    def get_processed_data(self):
        """
        Returns the processed data.

        Raises:
        ValueError: If data has not been loaded and processed.

        Returns:
        dict: Processed data including map patches and legends.
        """
        if not self.processed_data:
            raise ValueError("Data should be loaded and processed first")
        return self.processed_data


    def reconstruct_data(self, patches):
        """
        Reconstructs an image from overlapping patches, keeping the maximum value
        for overlapping pixels.
        
        Parameters:
        patches (numpy array): Image patches.
        
        Returns:
        numpy array: Reconstructed image.
        """
        assert self.overlap >= 0, "Overlap should be non-negative"
        step = patches.shape[2] - self.overlap
        img_shape = self.orig_size[::-1]  # Reverse the original size tuple to match the array shape
        img = np.zeros(img_shape)  # Initialize the image with zeros
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                x_start = i * step
                y_start = j * step
                x_end = min(x_start + patches.shape[2], img_shape[0])
                y_end = min(y_start + patches.shape[3], img_shape[1])
                
                img[x_start:x_end, y_start:y_end] = np.maximum(
                    img[x_start:x_end, y_start:y_end], 
                    patches[i, j, :x_end-x_start, :y_end-y_start]
                )
        
        return img


