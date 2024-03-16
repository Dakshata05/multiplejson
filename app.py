from flask import Flask , render_template,jsonify,request
from decode import sorted_alphanumeric,resizing_images, adding_padding, decoding_image
# from main import letterbox, model_loading,crop_random_part,sorted_alphanumeric
import subprocess,os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decode',methods=['GET'])
async def decode():
    folder_path = 'Images'
    # Check if folder path is provided
    if not folder_path:
        return jsonify({'error': 'Folder path not provided'}), 400

    # Check if the folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path provided'}), 400

    # Execute the decoding script
    subprocess.Popen(["python3", "decode.py", folder_path])

    return jsonify({'message': 'Decoding process started successfully'}), 200

@app.route('/process_images', methods=['GET'])
async def process_images():
    folder_path = 'Images'
    
    # Check if folder path is provided
    if not folder_path:
        return jsonify({'error': 'Folder path not provided'}), 400

    # Check if the folder exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid folder path provided'}), 400

    # Execute the image processing script
    subprocess.Popen(["python3", "main.py", folder_path])

    return jsonify({'message': 'Image processing started successfully'}), 200


if __name__ == '__main__':
    app.run(debug=True)
