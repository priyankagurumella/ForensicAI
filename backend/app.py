import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from phase1_fakenews.predict import predict_news
from phase2_deepfake.predict import predict_image

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "status": "ForensicAI API Running",
        "version": "1.0",
        "phases": ["Fake News Detection", "Deepfake Detection"]
    })

@app.route('/api/fakenews', methods=['POST'])
def fake_news():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_news(text)
    return jsonify(result)

@app.route('/api/deepfake', methods=['POST'])
def deepfake():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = request.files['image']
    image_bytes = image.read()
    result = predict_image(image_bytes)
    return jsonify(result)
@app.route('/api/scrape', methods=['POST'])
def scrape():
    from phase1_fakenews.scraper import scrape_article
    data = request.get_json()
    url = data.get('url', '')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    result = scrape_article(url)
    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True, port=5000)