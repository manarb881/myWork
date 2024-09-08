import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Configuration
DATASET_FOLDER = 'datasets/lfw'
EMBEDDINGS_FILE = 'models/lfw_embeddings.npy'
LABELS_FILE = 'models/lfw_labels.npy'
MODEL_FILE = 'models/Facenet/facenet_keras.h5'

# Load pre-trained FaceNet model
model = load_model(MODEL_FILE)

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
    person_names = os.listdir(dataset_folder)

    for label, person_name in enumerate(person_names):
        person_folder = os.path.join(dataset_folder, person_name)
        image_files = os.listdir(person_folder)
        
        for image_file in image_files:
            image_path = os.path.join(person_folder, image_file)
            image = load_image(image_path)
            embedding = image_to_embedding(image, model)
            embeddings.append(embedding)
            labels.append(label)
    
    return np.array(embeddings), np.array(labels), person_names

def save_embeddings(embeddings, labels):
    """Save embeddings and labels to files."""
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(LABELS_FILE, labels)

if __name__ == '__main__':
    embeddings, labels, person_names = generate_embeddings(DATASET_FOLDER)
    save_embeddings(embeddings, labels)
    print(f"Generated {len(embeddings)} embeddings and saved to {EMBEDDINGS_FILE} and {LABELS_FILE}.")
