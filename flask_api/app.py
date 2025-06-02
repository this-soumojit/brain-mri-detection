from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all logs, 1=INFO, 2=WARNING, 3=ERROR

import subprocess
import logging
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
import preprocess  # Your preprocess.py

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'nii', 'nii.gz', 'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'mriScan' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['mriScan']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File saved to {filepath}")

        try:
            img_array = preprocess.preprocess_image(filepath)

            # Convert back to PIL image for predict.py
            from PIL import Image
            import numpy as np

            img_processed = (img_array[0] * 255).astype('uint8')

            # If your array shape is (224,224,3), good.
            # If grayscale, convert accordingly.
            img_pil = Image.fromarray(img_processed)
            preprocessed_filename = f"preprocessed_{os.path.splitext(filename)[0]}.jpg"
            preprocessed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename)
            img_pil.save(preprocessed_filepath)
            logging.info(f"Preprocessed image saved to {preprocessed_filepath}")

        except Exception as e:
            logging.error(f"Preprocessing failed: {str(e)}")
            return jsonify({'error': 'Image preprocessing failed'}), 500

        prediction = run_prediction(preprocessed_filepath)

        if 'error' in prediction:
            return jsonify({'error': prediction['error']}), 500

        return jsonify({
            'success': True,
            'message': 'Prediction successful',
            'data': prediction
        })
    else:
        return jsonify({'error': 'Invalid file format'}), 400
def run_prediction(image_path):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        predict_script_path = os.path.abspath(os.path.join(current_dir, '..', 'model', 'predict.py'))

        if not os.path.isfile(predict_script_path):
            error_msg = f"predict.py not found at {predict_script_path}"
            logging.error(error_msg)
            return {'error': error_msg}

        logging.info(f"Running prediction script on: {image_path}")

        result = subprocess.run(
            ['python', predict_script_path, image_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True,
            timeout=60
        )

        logging.info(f"Prediction raw output:\n{result.stdout.strip()}")
        logging.info(f"Prediction error output:\n{result.stderr.strip()}")

        # Extract the last non-empty line from stdout assuming it contains JSON
        lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        if not lines:
            logging.error("No output from prediction script")
            return {'error': 'No output from prediction script'}

        json_line = lines[-1]  # last non-empty line (should be JSON)
        prediction_result = json.loads(json_line)

        return prediction_result

    except subprocess.CalledProcessError as e:
        logging.error(f"Prediction failed: {e.stderr}")
        return {'error': 'Prediction script execution failed'}
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding failed: {str(e)}")
        return {'error': 'Invalid JSON output from prediction script'}
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {'error': 'An unexpected error occurred during prediction'}

if __name__ == '__main__':
    app.run(debug=True)
