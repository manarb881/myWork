import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
# had l model makasho, n7awah m repo t3hom -m m4095m
from tensorflow.keras.models import load_model
from scipy.spatial import distance

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load pre-trained FaceNet model

# kifh 3rft bli na7awah? lokan tdkhli l repo te3 Facenet w trohi /src/models tsibi bli had el facenet_keras.h5 makasho, 7wsi 3la another one, Issam Menas tried Dlib and it was good -m m4095m 
model = load_model('models/Facenet/facenet_keras.h5')
database = np.load('models/lfw_embeddings.npy')
labels = np.load('models/lfw_labels.npy')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return image

def image_to_embedding(image, model):
    embedding = model.predict(image[np.newaxis, ...])
    embedding /= np.linalg.norm(embedding, ord=2)
    return embedding

def compare_embeddings(embedding_1, embedding_2, threshold=0.8):
    dist = np.linalg.norm(embedding_1 - embedding_2)
    return dist < threshold

def recognize_face(image, model, database, threshold=0.8):
    image_emb = image_to_embedding(image, model)
    min_dist = float('inf')
    name = "No Match Found"
    
    for label, embed in database.items():
        dist = np.linalg.norm(embed - image_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            name = label
            
    return name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = load_image(filepath)
        recognized_name = recognize_face(image, model, {label: embed for label, embed in zip(labels, database )})
        
        return render_template('index.html', result=recognized_name, image_url=url_for('static', filename='uploads/' + filename))
    
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
