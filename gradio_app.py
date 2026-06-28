import gradio as gr
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.phase1_fakenews.predict import predict_news
from backend.phase1_fakenews.scraper import scrape_article

def analyze_news(text, url):
    if url and url.startswith('http'):
        scraped = scrape_article(url)
        if 'error' not in scraped:
            text = scraped['text']
    if not text:
        return "Please enter text or URL", 0
    result = predict_news(text)
    verdict = f"{'⚠️ FAKE NEWS' if result['prediction'] == 'FAKE' else '✅ REAL NEWS'}"
    return verdict, result['confidence']

def analyze_deepfake(image):
    if image is None:
        return "Please upload an image", 0, 0, 0
    from backend.phase2_deepfake.predict import predict_image
    import io
    from PIL import Image
    img_bytes = io.BytesIO()
    Image.fromarray(image).save(img_bytes, format='JPEG')
    result = predict_image(img_bytes.getvalue())
    verdict = f"{'⚠️ DEEPFAKE' if result['prediction'] == 'FAKE' else '✅ REAL IMAGE'}"
    return verdict, result['confidence'], result['fake_prob'], result['real_prob']

news_interface = gr.Interface(
    fn=analyze_news,
    inputs=[
        gr.Textbox(label="Paste News Text", lines=5, placeholder="Paste news article here..."),
        gr.Textbox(label="Or Enter URL", placeholder="https://example.com/news")
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Confidence %")
    ],
    title="📰 Fake News Detection",
    description="Detect fake news with 99.6% accuracy"
)

deepfake_interface = gr.Interface(
    fn=analyze_deepfake,
    inputs=[gr.Image(label="Upload Image")],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Confidence %"),
        gr.Number(label="Fake Probability %"),
        gr.Number(label="Real Probability %")
    ],
    title="🖼️ Deepfake Detection",
    description="Detect deepfake images with 83.4% accuracy"
)

# Flask API for dashboard
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route('/api/fakenews', methods=['POST'])
def fake_news():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = predict_news(text)
    return jsonify(result)

@flask_app.route('/api/scrape', methods=['POST'])
def scrape():
    data = request.get_json()
    url = data.get('url', '')
    result = scrape_article(url)
    return jsonify(result)

@flask_app.route('/api/deepfake', methods=['POST'])
def deepfake_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    from backend.phase2_deepfake.predict import predict_image
    image = request.files['image']
    result = predict_image(image.read())
    return jsonify(result)

def run_flask():
    flask_app.run(host='0.0.0.0', port=7861, debug=False)

# Start Flask in background thread
threading.Thread(target=run_flask, daemon=True).start()

# Launch Gradio
demo = gr.TabbedInterface(
    [news_interface, deepfake_interface],
    ["📰 Fake News", "🖼️ Deepfake"]
)

demo.launch(server_name="0.0.0.0", server_port=7860)