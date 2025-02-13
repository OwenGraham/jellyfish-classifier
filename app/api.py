import os
import tempfile
from flask import Flask, request, jsonify

from app.inference import classify_jellyfish

app = Flask(__name__)

UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api', methods=['POST'])
def classify_jellyfish_api():
    file = request.files['image']

    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(temp_path)

    classification = classify_jellyfish(temp_path)

    return jsonify({'classification': classification})

if __name__ == '__main__':
    app.run(debug=True)