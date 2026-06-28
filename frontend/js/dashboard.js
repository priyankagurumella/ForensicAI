const API = 'https://priyankagurumella-forensicai.hf.space';

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
        const response = await fetch(`${API}/run/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data: [text, ''],
                fn_index: 0
            })
        });

        const data = await response.json();
        const verdict = data.data[0];
        const confidence = data.data[1];

        showNewsResult({
            prediction: verdict.includes('FAKE') ? 'FAKE' : 'REAL',
            confidence: confidence
        }, text);

        showToast('Analysis complete!', '✅');

    } catch (err) {
        showToast('Cannot connect to API!', '❌');
    } finally {
        document.getElementById('news-loading').style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '🔍 Analyze Article';
    }
}

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
        // Convert image to base64
        const base64 = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result.split(',')[1]);
            reader.readAsDataURL(file);
        });

        const response = await fetch(`${API}/run/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                data: [{ data: base64, is_file: false }],
                fn_index: 1
            })
        });

        const data = await response.json();
        const verdict = data.data[0];
        const confidence = data.data[1];
        const fake_prob = data.data[2];
        const real_prob = data.data[3];

        showDeepfakeResult({
            prediction: verdict.includes('DEEPFAKE') ? 'FAKE' : 'REAL',
            confidence: confidence,
            fake_prob: fake_prob,
            real_prob: real_prob
        }, file.name);

        showToast('Analysis complete!', '✅');

    } catch (err) {
        showToast('Cannot connect to API!', '❌');
    } finally {
        document.getElementById('deepfake-loading').style.display = 'none';
        btn.disabled = false;
        btn.innerHTML = '🔍 Analyze Image';
    }
}