# 🔍 ForensicAI — AI-Powered Fake News & Deepfake Detection System



![Python](https://img.shields.io/badge/Python-3.14-blue)




![Flask](https://img.shields.io/badge/Flask-3.0-green)




![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet-red)




![Accuracy](https://img.shields.io/badge/Accuracy-99.6%25-brightgreen)



> An end-to-end AI system that detects fake news using NLP and deepfake images using Computer Vision — with a professional web interface.

---

## 🚀 Live Demo

> Run locally — see setup instructions below

---

## 📸 Screenshots

### Fake News Detection


![Fake News](report/confusion_matrix.png)



### Deepfake Detection


![Deepfake](report/gradcam_output.png)



---

## 🧠 How It Works

### Phase 1 — Fake News Detection (NLP)
- Dataset: 44,000+ real news articles (Fake.csv + True.csv)
- Preprocessing: Text cleaning, stopword removal, lemmatization, TF-IDF vectorization
- Models trained & compared:
  - Passive Aggressive Classifier → **99.60% accuracy** ✅ (Best)
  - Logistic Regression → 99.11% accuracy
  - Naive Bayes → 94.72% accuracy
- Features: URL scraping, confidence scoring, word highlighting

### Phase 2 — Deepfake Detection (Computer Vision)
- Dataset: 10,000 real & AI-generated faces
- Model: EfficientNet-B0 (Transfer Learning, PyTorch)
- Training: 5 epochs on CPU → **83.41% validation accuracy**
- Features: Grad-CAM heatmap visualization, probability breakdown

---

## 🏗️ Project Structure

```
ForensicAI/
├── backend/
│   ├── app.py
│   ├── phase1_fakenews/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── scraper.py
│   ├── phase2_deepfake/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── gradcam.py
│   └── models/
├── frontend/
│   ├── index.html
│   ├── dashboard.html
│   ├── css/
│   └── js/
├── data/
├── report/
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/priyankagurumella/ForensicAI.git
cd ForensicAI
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- Fake News: [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Deepfake: [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

Place them in `data/raw/`

### 5. Train models
```bash
python backend/phase1_fakenews/train.py
python backend/phase2_deepfake/train.py
```

### 6. Run the app
```bash
python backend/app.py
```

Open `frontend/index.html` with Live Server

---

## 📊 Model Performance

| Module | Model | Accuracy | F1 Score |
|--------|-------|----------|----------|
| Fake News | Passive Aggressive | **99.60%** | 0.9959 |
| Fake News | Logistic Regression | 99.11% | 0.9908 |
| Fake News | Naive Bayes | 94.72% | 0.9454 |
| Deepfake | EfficientNet-B0 | **83.41%** | — |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-CORS |
| NLP | Scikit-learn, NLTK, TF-IDF |
| Computer Vision | PyTorch, EfficientNet-B0, Grad-CAM |
| Frontend | HTML5, CSS3, JavaScript |
| Data | Pandas, NumPy, Matplotlib, Seaborn |
| Scraping | BeautifulSoup, Requests |

---

## 🔮 Future Work

- 🎙️ Audio deepfake detection (voice cloning)
- 🌐 Browser extension for real-time detection
- 📱 Mobile app version
- 🤖 LLM integration for explanation generation
- 🔁 Active learning feedback loop

---

## 👩‍💻 Developer

**Durga Lalitha Priyanka Gurumella**
- 🎓 B.Tech Information Technology, MVGR College of Engineering
- 💼 Data Science Intern, Averixis Solutions
- 🐙 [GitHub](https://github.com/priyankagurumella)

---

## 📄 License

MIT License — feel free to use for educational purposes.