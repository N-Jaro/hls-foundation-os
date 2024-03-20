
import os
import gc
import rasterio
import numpy as np
from PIL import Image
import tensorflow as tf
from map_seg.data_utildata_util import DataLoader
from mmseg.apis import init_model, inference_model


def load_image_and_predict(map_file_name, prediction_path, model):
    """
    Load map image, find corresponding legend images, create inputs, predict, and reconstruct images.

    Parameters:
    map_file_name (str): Name of the map image file (e.g., 'AR_Maumee.tif').
    prediction_path (str): Path to save the predicted image.
    """
    # Set the paths
    map_dir = '/projects/bbym/shared/data/cma/validation/'
    map_img_path = os.path.join(map_dir, map_file_name)
    json_file_name = os.path.splitext(map_file_name)[0] + '.json'
    json_file_path = os.path.join(map_dir, json_file_name)

    patch_size=(224, 224, 3)
    overlap=30

    # Instantiate DataLoader and get processed data
    data_loader = DataLoader(map_img_path, json_file_path, patch_size, overlap)
    processed_data = data_loader.get_processed_data()
    
    map_patches = processed_data['map_patches']
    total_row, total_col, _, _, _, _ = map_patches.shape

        
    # # Define normalization constants
    # mean = tf.constant([0.485, 0.456, 0.406])
    # std = tf.constant([0.229, 0.224, 0.225])
    
    map_patches_reshape = map_patches.reshape(-1, *patch_size)

    # map_patches_reshape_norm = (map_patches_reshape - mean) / std

    # 'poly_legends', 'pt_legends', 'line_legends'
    for legend in ['poly_legends']:
        for legend_img, legend_label in processed_data[legend]:
            
            legend_patches = np.repeat(legend_img[np.newaxis],total_row*total_col, axis = 0)

            # legend_patches_norm = (legend_patches - mean) / std
            
            data_input = np.concatenate([map_patches_reshape, legend_patches], axis=-1)


            data_input_transpose = np.transpose(data_input, (2, 0, 1))

            
            predicted_patches = inference_model(model, data_input_transpose)

            print(predicted_patches.shape)
            
            predicted_patches_reshape = predicted_patches.reshape(total_row, total_col, patch_size[0], patch_size[1])
            
            reconstructed_image = data_loader.reconstruct_data(predicted_patches_reshape)
            
            # reconstructed_image = np.where(reconstructed_image >= 0.5, 1, 0).astype(np.uint8)

            output_image_path = os.path.join(prediction_path, f"{os.path.splitext(map_file_name)[0]}_{legend_label}.tif")

            with rasterio.open(map_img_path) as src:
                metadata = src.meta

            metadata.update({
                'dtype': 'uint8',
                'count': 1,
                'height': reconstructed_image.shape[0],
                'width': reconstructed_image.shape[1],
                'compress': 'lzw',
            })

            with rasterio.open(output_image_path, 'w', **metadata) as dst:
                dst.write(reconstructed_image*255, 1)
        
            gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

            print(f"Predicted image saved at: {output_image_path}")

prediction_path = '/projects/bbym/nathanj/hls-foundation-os/'
config_path = '/projects/bbym/nathanj/hls-foundation-os/configs/map_seg.py'
checkpoint_path = '/projects/bbym/nathanj/hls-foundation-os/save_model_1/best_mIoU_iter_9000.pth'

# init model and load checkpoint
model = init_model(config_path, checkpoint_path)

# Example of how to use the function
load_image_and_predict('KY_WestFranklin.tif', prediction_path, model)
