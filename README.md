# Properchy - Ames House Price Predictor

Properchy is a house price estimator for Ames, Iowa. It uses an XGBoost model trained on historical data (2006-2010) and features a frontend UI for real-time predictions. 

The project is built with **ZenML** and **MLflow** to ensure the entire modeling process—from data cleaning to deployment—is organized and reproducible.

---

## 🌟 Key Features

*   **Frontend UI**: A simple interface to get price predictions instantly.
*   **ML Pipeline**: Automated data cleaning, feature engineering, and outlier detection using ZenML.
*   **XGBoost Model**: Accuracy-focused predictions (~90.5% R²) using a gradient boosting regressor.
*   **Experiment Tracking**: Every training run is logged in MLflow with hyperparameters and metrics.
*   **Deployment Ready**: includes configuration files for Render, Railway, or VPS deployment.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
git clone https://github.com/yourusername/properchy.git
cd properchy
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Initialize ZenML & Stack
```bash
zenml init
zenml integration install mlflow -y

# Setup the MLflow stack
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow_deployer --flavor=mlflow
zenml stack register local_mlflow_stack -a default -o default -e mlflow_tracker -d mlflow_deployer
zenml stack set local_mlflow_stack
```

### 3. Run the App
To start the web interface locally:
```bash
python app.py
```
Visit `http://localhost:5000` to see the predictor.

---

## 🧪 Training & Pipelines

If you want to re-train the model:

*   **Train Model**: `python run_pipeline.py --model-type xgboost`
*   **Track Progress**: `mlflow ui --backend-store-uri file:./mlruns`
*   **Verify Accuracy**: Check `ml_pipeline` outputs in the ZenML dashboard (`zenml up`) or MLflow.

---

## 📁 Project Structure

*   `app.py`: Flask backend and prediction API.
*   `static/`: The frontend UI (HTML/CSS/JS).
*   `src/`: Core logic for data cleaning, feature engineering, and modeling.
*   `steps/`: Individual ZenML pipeline steps.
*   `pipelines/`: Training and deployment workflows.
*   `mlruns/`: Experiment logs and model artifacts.

---

## 📊 Model Performance

| Metric | Accuracy (R²) | Training Time |
| :--- | :--- | :--- |
| **XGBoost** | ~90.5% | < 10s |

---

## 🛠️ Deployment

This project includes configurations for deployment on **Render.com**. 
1. Push your code to GitHub (ensure `model.pkl` is tracked).
2. Connect your repo to Render.
3. Use the included `render.yaml` or set the start command to `gunicorn app:app`.

For a detailed guide, check [DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

**Happy Predicting! 🏠📊**
