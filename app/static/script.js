document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const btn = document.getElementById('submit-btn');
    const resultBox = document.getElementById('result-box');
    const resultText = document.getElementById('result-text');
    const resultDetails = document.getElementById('result-details');

    btn.textContent = 'Analyzing...';
    btn.disabled = true;

    const title = document.getElementById('title').value;
    const text = document.getElementById('text').value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, text })
        });
        const data = await response.json();
        
        btn.textContent = 'Predict';
        btn.disabled = false;

        if (response.ok) {
            resultBox.style.display = 'block';
            resultText.textContent = `Result: ${data.prediction}`;
            resultDetails.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}% (Time: ${data.processing_time_ms}ms)`;
        } else {
            alert(data.error || 'Failed to predict.');
        }
    } catch (err) {
        alert('Connection error.');
        btn.textContent = 'Predict';
        btn.disabled = false;
    }
});

let logInterval;
document.getElementById('retrain-btn').addEventListener('click', async function() {
    const btn = document.getElementById('retrain-btn');
    const consoleBox = document.getElementById('console-box');
    
    btn.disabled = true;
    btn.textContent = 'Retraining...';
    consoleBox.style.display = 'block';
    consoleBox.textContent = '> Training requested...\n';

    try {
        const response = await fetch('/train', { method: 'POST' });
        if (response.ok) {
            logInterval = setInterval(pollStatus, 2000);
        } else {
            btn.disabled = false;
            btn.textContent = 'Retrain Model';
            consoleBox.textContent += '> Error triggering build.\n';
        }
    } catch (err) {
        btn.disabled = false;
        btn.textContent = 'Retrain Model';
        consoleBox.textContent += '> Connection error.\n';
    }
});

async function pollStatus() {
    const consoleBox = document.getElementById('console-box');
    const btn = document.getElementById('retrain-btn');
    
    try {
        const response = await fetch('/train/status');
        const data = await response.json();
        
        if (data.logs) {
            consoleBox.textContent = data.logs.map(log => `> ${log}`).join('\n');
        }
        
        if (!data.in_progress) {
            clearInterval(logInterval);
            btn.disabled = false;
            btn.textContent = 'Retrain Model';
            consoleBox.textContent += '\n> Finished. Reloading...';
            setTimeout(() => location.reload(), 1500);
        }
    } catch (err) {
        console.error(err);
    }
}
