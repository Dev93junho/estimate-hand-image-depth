from tensorflow.keras.optimizers import Adam

def train_model(model, train_images, train_depth_maps, val_images, val_depth_maps):
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    model.fit(train_images, train_depth_maps, validation_data=(val_images, val_depth_maps), epochs=num_epochs, batch_size=batch_size)

