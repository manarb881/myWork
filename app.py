
import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
#from tensorflow.keras.models import load_model  
#from scipy.spatial import distance
from keras_facenet import FaceNet  # Use keras-facenet

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads' #Defines the directory where uploaded images will be stored.
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Configures the Flask app to use this directory for file uploads.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load pre-trained FaceNet model
# Load FaceNet model from keras-facenet
model = FaceNet().model
database = np.load('/Users/pc/myWork/models/lfw_embeddings.npy')
labels = np.load('/Users/pc/myWork/models/lfw_labels.npy')

def allowed_file(filename): # Checks if the file extension is allowed based on the ALLOWED_EXTENSIONS set by splitting in the dot 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (160, 160))
    image = image.astype(np.float32) / 255.0
    return image

def image_to_embedding(image, model):
    embedding = model.predict(image[np.newaxis, ...]) #here we augmented the size of the array because facenets always expects the output as batches even if its a single img 
    #exe: if an img is (160,160,3) its batch representation is (1,160,160,3)
    embedding /= np.linalg.norm(embedding, ord=2)
    return embedding

def compare_embeddings(embedding_1, embedding_2, threshold=0.8):
    dist = np.linalg.norm(embedding_1 - embedding_2)
    return dist < threshold

def recognize_face(image, model, database, threshold=0.8):
    image_emb = image_to_embedding(image, model)
    min_dist = float('inf')
    name = "No Match Found"
    
    for label, embed in database.items(): #database here is a dictionary
        dist = np.linalg.norm(embed - image_emb)
        if dist < min_dist and dist < threshold:
            min_dist = dist
            name = label
            
    return name

@app.route('/') #Defines the route for the homepage. When a user visits the root URL, it renders index.html.
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST']) #This line defines a route in your Flask web application. It maps the URL /upload to the upload_file() function.

def upload_file(): #This defines the function upload_file(), which handles the logic for uploading and processing a file (likely an image in our case).
    if 'file' not in request.files: #request.files is a dictionary-like object in Flask that contains all the files uploaded by the user via the form.
        return redirect(request.url) #If no file is found in the form data (i.e., 'file' not in request.files is True), the user is redirected back to the form (current URL) to retry.

    
    file = request.files['file'] #This line retrieves the uploaded file object from the form data under the key 'file' and stores it in the file variable.

    
    if file.filename == '': #This checks whether the uploaded file has a filename. If the filename is an empty string (i.e., the user selected no file), the function will return early.

        return redirect(request.url) #If no file was selected (i.e., the filename is empty), the user is redirected back to the form to upload a fil
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename) #This uses the secure_filename() function from the werkzeug package to sanitize the filename, ensuring it’s safe to use (e.g., removes special characters, spaces, or invalid filename characters).
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = load_image(filepath)
        recognized_name = recognize_face(image, model, {label: embed for label, embed in zip(labels, database)}) #pairs each label with its corresponding embedding, creating a dictionary where the label is the key and the embedding is the value.

        
        return render_template('index.html', result=recognized_name, image_url=url_for('static', filename='uploads/' + filename))
    
    return redirect(request.url)

if __name__ == '__main__': #Ensures that the following code runs only if the script is executed directly.
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER) #Creates the upload directory if it doesn’t exist.
    app.run(debug=True) # Starts the Flask development server with debugging enabled.
