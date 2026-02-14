// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const predictBtn = document.getElementById('predictBtn');
const resultDisplay = document.getElementById('resultDisplay');
const loadingOverlay = document.getElementById('loadingOverlay');
const priceAmount = document.getElementById('priceAmount');
const resetBtn = document.getElementById('resetBtn');

// Form submission handler
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = new FormData(predictionForm);
    const data = {};
    
    formData.forEach((value, key) => {
        data[key] = isNaN(value) ? value : Number(value);
    });
    
    // Show loading
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result.predicted_price);
        } else {
            showError(result.error || 'Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to server. Please try again.');
    } finally {
        hideLoading();
    }
});

// Display prediction result
function displayResult(price) {
    // Hide form, show result
    predictionForm.style.display = 'none';
    resultDisplay.style.display = 'block';
    
    // Animate price
    animateValue(priceAmount, 0, price, 800);
}

// Animate number counting
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (range * easeOut);
        
        element.textContent = formatPrice(current);
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = formatPrice(end);
        }
    }
    
    requestAnimationFrame(update);
}

// Format price
function formatPrice(price) {
    return Math.round(price).toLocaleString('en-US');
}

// Show loading
function showLoading() {
    loadingOverlay.style.display = 'flex';
    predictBtn.disabled = true;
}

// Hide loading
function hideLoading() {
    loadingOverlay.style.display = 'none';
    predictBtn.disabled = false;
}

// Show error
function showError(message) {
    alert(`Error: ${message}`);
}

// Reset button
resetBtn.addEventListener('click', () => {
    resultDisplay.style.display = 'none';
    predictionForm.style.display = 'block';
    predictionForm.reset();
});

// (Removed aggressive input validation to improve user typing experience)

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        
        if (data.model_loaded) {
            console.log('✓ Ready');
        }
    } catch (error) {
        console.error('✗ API error:', error);
    }
}

// Initialize
checkHealth();
