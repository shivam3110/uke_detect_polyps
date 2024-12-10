from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os
import subprocess
import sys
from pathlib import Path
import json

app = Flask(__name__)

# Set base directory for all operations
BASE_DIR = Path(__file__).resolve().parent

# Configuring directories
app.config['UPLOAD_FOLDER'] = BASE_DIR / 'data' / 'uploads'
app.config['ANNOTATED_FRAME_DIR'] = BASE_DIR / 'data' / 'annotated_frames'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATED_FRAME_DIR'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400
        if not file.filename.endswith(('.mp4', '.avi')):
            return jsonify({'error': 'Invalid file type'}), 400
        
        filename = file.filename
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'path': filepath.as_posix()})
    return render_template('upload.html')


@app.route('/execute-model', methods=['POST'])
def execute_model():
    data = request.get_json()
    video_path = app.config['UPLOAD_FOLDER'] / Path(data['videoPath']).name
    model_type = data['modelType']
    script_path = BASE_DIR / "models" / f"{model_type}_eval.py"

    if not script_path.exists():
        return jsonify({'error': 'Model script not found'}), 404

    # Execute the model-specific script
    process_cmd = [sys.executable, script_path.as_posix(), video_path.as_posix()]
    result = subprocess.run(process_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        images = list(app.config['ANNOTATED_FRAME_DIR'].glob('*.jpg'))
        images_url = [url_for('send_image', filename=img.name) for img in images]
        # Attempt to load corresponding JSON files to count bounding boxes
        polyp_count = 0
        for img in images:
            json_path = img.with_suffix('.json')  # Correct way to change the suffix
            if json_path.exists():
                with open(json_path, 'r') as file:
                    data = json.load(file)
                    if 'bbox' in data and data['bbox']:  # Check if 'bbox' key exists and has data
                        polyp_count += 1
        return jsonify({'result': 'Processing complete', 'images': images_url, 'polyp_count': polyp_count})
    else:
        return jsonify({'error': 'Model execution failed', 'details': result.stderr.decode()}), 500

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(app.config['ANNOTATED_FRAME_DIR'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
