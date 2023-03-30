import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from config import input_height, input_width, num_channels, batch_size, num_epochs
from data_utils import load_data, preprocess_data
from model import build_model
from tensorflow_model_optimization.sparsity import keras as sparsity
import argparse
from datetime import datetime

def apply_pruning(model):
    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.0,
                                                     final_sparsity=0.5,
                                                     begin_step=2000,
                                                     end_step=10000)
    }

    pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
    return pruned_model

def apply_quantization(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    with open('quantized_model.tflite', 'wb') as f:
        f.write(quantized_model)

    return quantized_model

def train_model(model, train_images, train_depth_maps, val_images, val_depth_maps):
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    model.fit(train_images, train_depth_maps, validation_data=(val_images, val_depth_maps), epochs=num_epochs, batch_size=batch_size)

def main(apply_pruning_flag, apply_quantization_flag):
    image_dir = 'path/to/image/directory'
    depth_map_dir = 'path/to/depth_map/directory'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    images, depth_maps = load_data(image_dir, depth_map_dir)
    train_images, val_images, train_depth_maps, val_depth_maps = preprocess_data(images, depth_maps, input_height, input_width)

    model = build_model(input_height, input_width, num_channels)
    train_model(model, train_images, train_depth_maps, val_images, val_depth_maps)

    # Save the original model for later use
    model.save(f'hand_depth_model_{timestamp}.h5')
    
    if apply_pruning_flag:
        # Apply pruning
        pruned_model = apply_pruning(model)
        train_model(pruned_model, train_images, train_depth_maps, val_images, val_depth_maps)
        pruned_model.save(f'pruned_hand_depth_model_{timestamp}.h5')

    if apply_quantization_flag:
        # Apply quantization
        quantized_model = apply_quantization(model)
        with open(f'quantized_model_{timestamp}.tflite', 'wb') as f:
            f.write(quantized_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a depth estimation model with optional pruning and quantization.')
    parser.add_argument('--prune', action='store_true', help='Apply pruning to the model')
    parser.add_argument('--quantize', action='store_true', help='Apply quantization to the model')
    args = parser.parse_args()

    main(args.prune, args.quantize)
