from config import input_height, input_width, num_channels, batch_size, num_epochs
from data_utils import load_data, preprocess_data
from model import build_model
from train import train_model

def main():
    image_dir = 'path/to/image/directory'
    depth_map_dir = 'path/to/depth_map/directory'

    images, depth_maps = load_data(image_dir, depth_map_dir)
    train_images, val_images, train_depth_maps, val_depth_maps = preprocess_data(images, depth_maps, input_height, input_width)

    model = build_model(input_height, input_width, num_channels)
    train_model(model, train_images, train_depth_maps, val_images, val_depth_maps)

    # Save the model for later use
    model.save('hand_depth_model.h5')

if __name__ == '__main__':
    main()
