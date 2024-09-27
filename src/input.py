import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Define image size
IMAGE_HEIGHT, IMAGE_WIDTH = 128, 128

# Load the trained model
model = load_model('lane_segmentation_model.keras')

# Function to preprocess the input image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to post-process the predicted output
def post_process_output(prediction):
    # Assuming binary segmentation, threshold the prediction
    prediction = (prediction > 0.5).astype(np.uint8)  # Apply threshold
    prediction = np.squeeze(prediction)  # Remove batch dimension
    return prediction * 255  # Scale to 0-255 for visualization

# Function to predict lane lines
def predict_lane_line(image_path):
    # Preprocess the input image
    processed_image = preprocess_image(image_path)

    # Make predictions
    prediction = model.predict(processed_image)

    # Post-process the prediction
    output_mask = post_process_output(prediction)

    # Display the output
    cv2.imshow('Input Image', cv2.imread(image_path))
    cv2.imshow('Predicted Lane Line', output_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output mask
    output_filename = 'predicted_lane_line.png'
    cv2.imwrite(output_filename, output_mask)
    print(f"Output saved as: {output_filename}")

# Example usage
input_image_path = r'D:\road lane line detection\data_road\testing\image_2\um_000004.png'  # Replace with your input image path
predict_lane_line(input_image_path)
