const API = 'http://127.0.0.1:5000';
let newsHistory = [];
let deepfakeHistory = [];

// ===== TOAST NOTIFICATION =====
function showToast(message, emoji = '✅') {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${emoji}</span> ${message}`;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ===== TAB SWITCHING =====
function showTab(tab) {
    document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
    document.querySelectorAll('.sidebar-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(tab + '-tab').style.display = 'block';
    event.target.closest('.sidebar-btn').classList.add('active');
}

// ===== INPUT SWITCHING =====
function switchInput(type) {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById('text-input').style.display = type === 'text' ? 'block' : 'none';
    document.getElementById('url-input').style.display = type === 'url' ? 'block' : 'none';
}

// ===== IMAGE PREVIEW =====
function previewImage(event) {
    const file = event.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('preview-img').src = e.target.result;
        document.getElementById('image-preview').style.display = 'block';
        document.getElementById('deepfake-btn').style.display = 'inline-flex';
        document.getElementById('upload-area').style.display = 'none';
        showToast('Image loaded successfully!', '🖼️');
    };
    reader.readAsDataURL(file);
}

// ===== DRAG AND DROP =====
function setupDragDrop() {
    const uploadArea = document.getElementById('upload-area');
    if (!uploadArea) return;

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            document.getElementById('image-input').files = e.dataTransfer.files;
            previewImage({ target: { files: [file] } });
        }
    });
}

// ===== CHARACTER COUNTER =====
function setupCharCounter() {
    const textarea = document.getElementById('news-text');
    const counter = document.getElementById('char-counter');
    if (!textarea || !counter) return;
    textarea.addEventListener('input', () => {
        counter.textContent = `${textarea.value.length} characters`;
    });
}

// ===== WORD HIGHLIGHTING =====
function highlightWords(text) {
    const suspiciousWords = [
        'shocking', 'breaking', 'exclusive', 'secret', 'hoax',
        'conspiracy', 'fake', 'fraud', 'lie', 'exposed',
        'bombshell', 'scandal', 'urgent', 'alert', 'warning',
        'unbelievable', 'miracle', 'cure', 'banned', 'censored',
        'they dont want', 'hidden truth', 'cover up', 'cover-up'
    ];
    const words = text.split(' ');
    return words.map(word => {
        const clean = word.toLowerCase().replace(/[^a-z]/g, '');
        if (suspiciousWords.includes(clean)) {
            return `<span class="word-fake">${word}</span>`;
        }
        return `<span class="word-neutral">${word}</span>`;
    }).join(' ');
}

// ===== MODEL COMPARISON CHART =====
function renderChart(containerId) {
    const models = [
        { name: 'Passive Aggressive', acc: 99.6 },
        { name: 'Logistic Regression', acc: 99.11 },
        { name: 'Naive Bayes', acc: 94.72 }
    ];

    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
        <div class="chart-title">Model Comparison</div>
        ${models.map(m => `
            <div class="chart-row">
                <div class="chart-label">${m.name}</div>
                <div class="chart-bar-wrap">
                    <div class="chart-bar-fill" data-width="${m.acc}"></div>
                </div>
                <div class="chart-pct">${m.acc}%</div>
            </div>
        `).join('')}
    `;

    setTimeout(() => {
        container.querySelectorAll('.chart-bar-fill').forEach(bar => {
            bar.style.width = bar.dataset.width + '%';
        });
    }, 200);
}

// ===== UPDATE HISTORY =====
function updateHistory(type, text, prediction, confidence) {
    const list = type === 'news' ? newsHistory : deepfakeHistory;
    list.unshift({ text, prediction, confidence });
    if (list.length > 5) list.pop();

    const containerId = type === 'news' ? 'news-history-list' : 'deepfake-history-list';
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = list.map(item => `
        <div class="history-item">
            <div class="history-text">${item.text.substring(0, 70)}...</div>
            <div style="display:flex; gap:8px; align-items:center; flex-shrink:0">
                <span style="font-size:0.75rem; color:rgba(255,255,255,0.25)">${item.confidence}%</span>
                <span class="history-badge ${item.prediction === 'FAKE' ? 'badge-fake' : 'badge-real'}">
                    ${item.prediction}
                </span>
            </div>
        </div>
    `).join('');
}

// ===== ANALYZE NEWS =====
async function analyzeNews() {
    const textInput = document.getElementById('news-text').value;
    const urlInput = document.getElementById('news-url') ?
        document.getElementById('news-url').value : '';
    const text = textInput || urlInput;

    if (!text.trim()) {
        showToast('Please enter some text or URL!', '⚠️');
        return;
    }

    const btn = document.querySelector('#fakenews-tab .analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    document.getElementById('news-loading').style.display = 'block';
    document.getElementById('news-result').style.display = 'none';

    try {
        let bodyText = text;

        if (text.startsWith('http')) {
            showToast('Scraping article...', '🌐');
            const scrapeRes = await fetch(`${API}/api/scrape`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: text })
            });
            const scrapeData = await scrapeRes.json();
            if (scrapeData.error) {
                showToast('Could not scrape URL', '❌');
                return;
            }
            bodyText = scrapeData.text || text;
        }

        const response = await fetch(`${API}/api/fakenews`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: bodyText })
        });

        const data = await response.json();
        showNewsResult(data, bodyText);
        showToast('Analysis complete!', '✅');

    } catch (err) {
        showToast('Cannot connect to API. Start Flask!', '❌');
    } finally {
        document.getElementById('news-loading').style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '🔍 Analyze Article';
    }
}

// ===== SHOW NEWS RESULT =====
function showNewsResult(data, originalText) {
    const isFake = data.prediction === 'FAKE';
    const conf = data.confidence;

    const card = document.querySelector('#news-result .result-card');
    card.className = `result-card ${isFake ? 'result-fake' : 'result-real'}`;

    document.getElementById('news-verdict').innerHTML =
        `<span class="${isFake ? 'verdict-fake' : 'verdict-real'}">
            ${isFake ? '⚠️ FAKE NEWS DETECTED' : '✅ REAL NEWS VERIFIED'}
        </span>`;

    document.getElementById('news-confidence').innerHTML = `
        <div class="speedometer">
            <div class="speed-value">${conf}</div>
            <div class="speed-pct">%</div>
            <div class="speed-label">Confidence Score</div>
        </div>
    `;

    const bar = document.getElementById('news-bar');
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = conf + '%';
        bar.className = `confidence-fill ${isFake ? 'fill-fake' : 'fill-real'}`;
    }, 100);

    const highlighted = highlightWords(originalText.substring(0, 400));
    document.getElementById('news-details').innerHTML = `
        <strong>Suspicious Words Detected:</strong>
        <div class="word-highlight">${highlighted}...</div>
        <br>
        <strong>Model:</strong> Passive Aggressive Classifier &nbsp;|&nbsp;
        <strong>Training Accuracy:</strong> 99.6% &nbsp;|&nbsp;
        <strong>Dataset:</strong> 44,000+ Real News Articles
    `;

    document.getElementById('news-result').style.display = 'block';
    renderChart('news-chart');
    updateHistory('news', originalText, data.prediction, conf);
}

// ===== ANALYZE DEEPFAKE =====
async function analyzeDeepfake() {
    const file = document.getElementById('image-input').files[0];
    if (!file) {
        showToast('Please upload an image!', '⚠️');
        return;
    }

    const btn = document.getElementById('deepfake-btn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    document.getElementById('deepfake-loading').style.display = 'block';
    document.getElementById('deepfake-result').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('image', file);

        const response = await fetch(`${API}/api/deepfake`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        showDeepfakeResult(data, file.name);
        showToast('Analysis complete!', '✅');

    } catch (err) {
        showToast('Cannot connect to API. Start Flask!', '❌');
    } finally {
        document.getElementById('deepfake-loading').style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '🔍 Analyze Image';
    }
}

// ===== SHOW DEEPFAKE RESULT =====
function showDeepfakeResult(data, filename) {
    const isFake = data.prediction === 'FAKE';
    const conf = data.confidence;

    const card = document.querySelector('#deepfake-result .result-card');
    card.className = `result-card ${isFake ? 'result-fake' : 'result-real'}`;

    document.getElementById('deepfake-verdict').innerHTML =
        `<span class="${isFake ? 'verdict-fake' : 'verdict-real'}">
            ${isFake ? '⚠️ DEEPFAKE DETECTED' : '✅ REAL IMAGE VERIFIED'}
        </span>`;

    document.getElementById('deepfake-confidence').innerHTML = `
        <div class="speedometer">
            <div class="speed-value">${conf}</div>
            <div class="speed-pct">%</div>
            <div class="speed-label">Confidence Score</div>
        </div>
    `;

    const bar = document.getElementById('deepfake-bar');
    bar.style.width = '0%';
    setTimeout(() => {
        bar.style.width = conf + '%';
        bar.className = `confidence-fill ${isFake ? 'fill-fake' : 'fill-real'}`;
    }, 100);

    document.getElementById('prob-breakdown').innerHTML = `
        <div class="prob-item">
            <div class="prob-label">FAKE Probability</div>
            <div class="prob-value" style="color:#ef4444">${data.fake_prob}%</div>
        </div>
        <div class="prob-item">
            <div class="prob-label">REAL Probability</div>
            <div class="prob-value" style="color:#22c55e">${data.real_prob}%</div>
        </div>
    `;

    document.getElementById('deepfake-result').style.display = 'block';
    updateHistory('deepfake', filename, data.prediction, conf);
}

// ===== INIT =====
window.addEventListener('load', () => {
    setupDragDrop();
    setupCharCounter();
    renderChart('news-chart');
});