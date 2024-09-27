import os
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define image size
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128

# Load and preprocess images and corresponding ground truths
def load_images_and_masks(image_folder, mask_folder):
    images, masks = [], []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')  # Add valid image extensions

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_folder, filename)
            mask_filename = filename.replace('um_', 'um_lane_')  # Replace 'um_' with 'um_lane_'
            mask_path = os.path.join(mask_folder, mask_filename)

            # Verify if the image and mask paths exist
            if not os.path.exists(image_path):
                print(f"Image does not exist: {image_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"Mask does not exist: {mask_path}")
                continue

            print(f"Reading: {image_path}")
            img = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is not None and mask is not None:
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
                # Normalize image and mask
                img = img / 255.0
                mask = mask / 255.0
                mask = np.expand_dims(mask, axis=-1)  # Make it a single-channel image
                images.append(img)
                masks.append(mask)
            else:
                print(f"Error reading image or mask for {filename}")
        else:
            print(f"Skipping non-image file: {filename}")
    
    print(f"Total images loaded: {len(images)}")
    print(f"Total masks loaded: {len(masks)}")

    return np.array(images), np.array(masks)

# Build U-Net model
def unet_model():
    inputs = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    
    # Encoding path
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    
    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    
    # Decoding path
    u1 = UpSampling2D((2, 2))(c3)
    concat1 = concatenate([u1, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
    
    u2 = UpSampling2D((2, 2))(c4)
    concat2 = concatenate([u2, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(concat2)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)  # Binary segmentation
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Save the model as .h5
def train_and_save_model(images, masks):
    model = unet_model()
    
    # Checkpoint to save model
    checkpoint = ModelCheckpoint('lane_segmentation_model.keras', save_best_only=False, monitor='val_loss', mode='min')
    
    print("Training data shape:", images.shape)
    print("Masks shape:", masks.shape)

    if images.shape[0] == 0 or masks.shape[0] == 0:
        raise ValueError("No training data available.")

    # Train the model
    history = model.fit(images, masks, epochs=10, batch_size=8, validation_split=0.2, callbacks=[checkpoint])


# Fix paths using raw string literals or forward slashes
image_folder = r'data_road\training\image_2'
mask_folder = r'data_road\training\gt_image_2'



# Load the images and masks
images, masks = load_images_and_masks(image_folder, mask_folder)
print("Images shape:", images.shape)
print("Masks shape:", masks.shape)

# Check if images and masks contain samples
if images.shape[0] == 0 or masks.shape[0] == 0:
    raise ValueError("No training data available.")


# Train the model and save it as .h5
train_and_save_model(images, masks)
