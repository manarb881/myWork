import os
import numpy as np
import cv2
from keras_facenet import FaceNet

# Configuration
DATASET_FOLDER = '/Users/pc/myWork/datasets/lfw'
EMBEDDINGS_FILE = '/Users/pc/myWork/models/lfw_embeddings.npy'
LABELS_FILE = '/Users/pc/myWork/models/lfw_labels.npy'

# Load pre-trained FaceNet model
model = FaceNet().model

def load_image(image_path):
    """Load and preprocess an image."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return image

def image_to_embedding(image, model):
    """Convert an image to a face embedding."""
    embedding = model.predict(image[np.newaxis, ...])
    embedding /= np.linalg.norm(embedding, ord=2)
    return embedding 

def generate_embeddings(dataset_folder):
    """Generate embeddings for all images in the dataset folder."""
    embeddings = []
    labels = []
    
    # Adjusted folder structure
    deep_funneled_folder = os.path.join(dataset_folder, 'deep-funneled', 'deep-funneled')
    person_names = os.listdir(deep_funneled_folder)

    print(f"Found persons: {person_names}")

    for person_name in person_names:
        person_folder = os.path.join(deep_funneled_folder, person_name)
        
        # Ensure person_folder is a directory
        if not os.path.isdir(person_folder):
            print(f"Skipping {person_folder} as it is not a directory.")
            continue

        image_files = [f for f in os.listdir(person_folder) if os.path.isfile(os.path.join(person_folder, f))]
        print(f"Processing folder: {person_folder}, Images: {image_files}")

        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            image = load_image(image_path)
            embedding = image_to_embedding(image, model)
            embeddings.append(embedding)
            labels.append(person_name)  # Store person name as label

    print(f"Generated embeddings: {len(embeddings)}")
    return np.array(embeddings), np.array(labels), person_names

def save_embeddings(embeddings, labels):
    """Save embeddings and labels to files."""
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(LABELS_FILE, labels)

if __name__ == '__main__':
    embeddings, labels, person_names = generate_embeddings(DATASET_FOLDER)
    save_embeddings(embeddings, labels)
    print(f"Generated {len(embeddings)} embeddings and saved to {EMBEDDINGS_FILE} and {LABELS_FILE}.")

