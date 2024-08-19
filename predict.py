import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

def process_image(image):
    """Preprocess the image for model prediction."""
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k=5):
    """Predict the class and probabilities of an image."""
    im = Image.open(image_path)
    processed_image = process_image(np.asarray(im))
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_k_probs, top_k_classes = tf.nn.top_k(predictions[0], k=top_k)
    return top_k_probs.numpy(), top_k_classes.numpy()

def load_class_names(json_path):
    """Load class names from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Predict image class using a Keras model.')
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model.')
    parser.add_argument('--json', type=str, help='Path to JSON file for class name mapping.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to display.')

    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Lambda})

    # Predict
    probs, classes = predict(args.image_path, model, top_k=args.top_k)

    # Load class names if provided
    class_names = None
    if args.json:
        class_names = load_class_names(args.json)

    # Print the top K classes and their probabilities
    for i in range(args.top_k):
        class_label = str(classes[i])
        class_name = class_names.get(class_label, class_label) if class_names else class_label
        print(f"Class: {class_name}, Probability: {probs[i]:.4f}")

if __name__ == "__main__":
    main()
