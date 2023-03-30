import argparse
import os
from datetime import datetime
from config import input_height, input_width, num_channels, batch_size, num_epochs
from data_utils import load_data, preprocess_data
from model import build_model
from train import train_model
from pruning import apply_pruning
from quantization import apply_quantization

def main():
    parser = argparse.ArgumentParser(description='Depth Estimation Model Training and Conversion')
    parser.add_argument('--prune', action='store_true', help='Apply pruning')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization')
    args = parser.parse_args()

    image_dir = 'path/to/image/directory'
    depth_map_dir = 'path/to/depth_map/directory'

    images, depth_maps = load_data(image_dir, depth_map_dir)
    train_images, val_images, train_depth_maps, val_depth_maps = preprocess_data(images, depth_maps, input_height, input_width)

    model = build_model(input_height, input_width, num_channels)
    train_model(model, train_images, train_depth_maps, val_images, val_depth_maps)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f'models/hand_depth_model_{timestamp}', save_format='tf')

    if args.prune:
        pruned_model = apply_pruning(model)
        train_model(pruned_model, train_images, train_depth_maps, val_images, val_depth_maps)
        pruned_model.save(f'models/pruned_hand_depth_model_{timestamp}', save_format='tf')

    if args.quantize:
        quantized_model = apply_quantization(model)
        with open(f'models/quantized_model_{timestamp}.tflite', 'wb') as f:
            f.write(quantized_model)

if __name__ == '__main__':
    main()
