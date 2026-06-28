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
    confidence = result['confidence']
    return verdict, confidence

def analyze_deepfake(image):
    if image is None:
        return "Please upload an image", 0
    
    from backend.phase2_deepfake.predict import predict_image
    import io
    from PIL import Image
    
    img_bytes = io.BytesIO()
    Image.fromarray(image).save(img_bytes, format='JPEG')
    result = predict_image(img_bytes.getvalue())
    
    verdict = f"{'⚠️ DEEPFAKE' if result['prediction'] == 'FAKE' else '✅ REAL IMAGE'}"
    return verdict, result['confidence'], result['fake_prob'], result['real_prob']

# Fake News Interface
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

# Deepfake Interface
deepfake_interface = gr.Interface(
    fn=analyze_deepfake,
    inputs=[
        gr.Image(label="Upload Image")
    ],
    outputs=[
        gr.Textbox(label="Verdict"),
        gr.Number(label="Confidence %"),
        gr.Number(label="Fake Probability %"),
        gr.Number(label="Real Probability %")
    ],
    title="🖼️ Deepfake Detection",
    description="Detect deepfake images with 83.4% accuracy"
)

# Combined App
demo = gr.TabbedInterface(
    [news_interface, deepfake_interface],
    ["📰 Fake News", "🖼️ Deepfake"]
)

if __name__ == "__main__":
    demo.launch()