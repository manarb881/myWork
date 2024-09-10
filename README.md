# Face Recognition App

A simple web app for face recognition using Flask. Upload an image and get recognition results.

## Features

- Upload an image for recognition.
- View the recognition result and uploaded image.

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/face-recognition-app.git
    cd face-recognition-app
    ```

2. **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use .venv\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the model**: pip install keras facenet

5. **Generate embeddings:**

    ```bash
    python generate_embeddings.py
    ```

6. **Run the app:**

    ```bash
    python app.py
    ```

    Visit `http://127.0.0.1:5000/` in your browser.

## Project Structure

- `app.py` – Flask app script.
- `generate_embeddings.py` – Script to generate face embeddings.
- `templates/index.html` – HTML template for the app.
- `static/uploads/` – Directory for uploaded images.
- `models/` – Directory for face embeddings and model files.

## License

MIT License. See the [LICENSE](LICENSE) file for details.

