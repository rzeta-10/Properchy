# Properchy - House Price Prediction System

A production-grade machine learning system for predicting house prices using MLflow and ZenML. This project implements various design patterns and best practices for building maintainable and scalable ML pipelines.

## 🌟 Features

- 🏗️ End-to-end ML pipeline with ZenML orchestration
- 📊 Comprehensive data analysis and feature engineering
- 🤖 Multiple model support (XGBoost, Linear Regression)
- 📈 MLflow experiment tracking and model registry
- 🚀 Automated model deployment with MLflow
- ♻️ Continuous deployment pipeline
- 🔄 Real-time inference API
- 📱 Interactive UI for experiment tracking

## 🏛️ Architecture

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐
│  Data Ingestion │
└────────┬────────┘
         │
┌────────▼────────┐
│  Preprocessing  │
└────────┬────────┘
         │
┌────────▼────────┐
│ Model Training  │
└────────┬────────┘
         │
┌────────▼────────┐
│   Evaluation    │
└────────┬────────┘
         │
┌────────▼────────┐
│   Deployment    │
└─────────────────┘
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/properchy.git
cd properchy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize ZenML

```bash
# Initialize ZenML repository
zenml init

# Install MLflow integration
zenml integration install mlflow -y

# Register MLflow components
zenml experiment-tracker register mlflow_tracker --flavor=mlflow --tracking_uri=file:./mlruns
zenml model-deployer register mlflow_deployer --flavor=mlflow

# Create and set stack
zenml stack register local_mlflow_stack -a default -o default -e mlflow_tracker -d mlflow_deployer
zenml stack set local_mlflow_stack
```

### 3. Run Training Pipeline

```bash
# Train with XGBoost (default)
python run_pipeline.py

# Train with Linear Regression
python run_pipeline.py --model-type linear_regression
```

## 📊 UI Setup and Access

### MLflow UI

MLflow provides a web interface for tracking experiments, comparing runs, and managing models.

**Start MLflow UI:**
```bash
mlflow ui --backend-store-uri file:./mlruns
```

**Access the UI:**
- Open your browser and navigate to: `http://localhost:5000`
- Default port is 5000, use `--port` to change it

**Features:**
- 📈 View experiment metrics and parameters
- 📊 Compare multiple model runs
- 🔍 Inspect model artifacts and logs
- 📦 Manage model registry and versions

### ZenML Dashboard (Optional)

For advanced pipeline monitoring and management:

```bash
# Install ZenML server (if not already installed)
pip install "zenml[server]"

# Start ZenML server
zenml up

# Access dashboard at http://localhost:8237
```

**Dashboard Features:**
- 🔄 Monitor pipeline runs in real-time
- 📋 View step-by-step execution details
- 🗂️ Browse artifacts and metadata
- 📊 Visualize pipeline DAGs

## 🎯 Usage Guide

### Training Pipeline

The training pipeline includes:
1. Data ingestion from ZIP archive
2. Missing value imputation
3. Feature engineering (log transformation)
4. Outlier detection and removal
5. Train-test split
6. Model training with preprocessing
7. Model evaluation

**Run training:**
```bash
python run_pipeline.py --model-type xgboost
```

### Deployment Pipeline

Deploy trained models for inference:

```bash
# Deploy the model
python run_deployment.py

# Deploy with specific model type
python run_deployment.py --model-type linear_regression

# Stop deployment service
python run_deployment.py --stop-service
```

### Making Predictions

```bash
# Run sample predictions
python sample_predict.py
```

## 📁 Project Structure

```
properchy/
├── analysis/                   # Data analysis notebooks and scripts
│   └── analysis_src/          # Analysis modules
├── config/                     # Configuration files
├── data/                       # Data directory
│   └── archive.zip            # Training data
├── pipelines/                  # Pipeline definitions
│   ├── training_pipeline.py   # ML training pipeline
│   └── deployment_pipeline.py # Deployment pipeline
├── src/                       # Core source code
│   ├── data_splitter.py       # Data splitting logic
│   ├── feature_engineering.py # Feature transformations
│   ├── handle_missing_values.py
│   ├── ingest_data.py         # Data ingestion
│   ├── model_building.py      # Model strategies
│   ├── model_evaluator.py     # Evaluation metrics
│   └── outlier_detection.py   # Outlier handling
├── steps/                     # ZenML pipeline steps
│   ├── data_ingestion_step.py
│   ├── data_splitter_step.py
│   ├── feature_engineering_step.py
│   ├── handle_missing_values_step.py
│   ├── model_building_step.py
│   ├── model_evaluator_step.py
│   ├── outlier_detection_step.py
│   ├── dynamic_importer.py
│   ├── model_loader.py
│   ├── prediction_service_loader.py
│   └── predictor.py
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
├── run_pipeline.py            # Training script
├── run_deployment.py          # Deployment script
├── sample_predict.py          # Prediction example
└── README.md                  # This file
```

## 🔧 Configuration

### Model Selection

The system supports multiple model types:

**XGBoost (Recommended)**
- Gradient boosting algorithm
- Better performance on complex patterns
- Handles non-linear relationships well
- Default choice for production

**Linear Regression**
- Simple and interpretable
- Fast training and inference
- Good baseline model
- Useful for understanding feature importance

### Pipeline Configuration

Edit `config.yaml` to customize:
- Data paths
- Model hyperparameters
- Feature engineering strategies
- Evaluation metrics

## 📈 Model Performance

Current benchmark results:

| Model | MSE | R² Score | Training Time |
|-------|-----|----------|---------------|
| XGBoost | 0.0134 | 0.9048 | ~8s |
| Linear Regression | TBD | TBD | ~2s |

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. ZenML Stack Not Found**
```bash
# Reinitialize ZenML
zenml init
zenml stack set local_mlflow_stack
```

**3. MLflow Tracking URI Error**
```bash
# Verify MLflow directory exists
ls mlruns/

# Restart MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

**4. Port Already in Use**
```bash
# Use different port for MLflow
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=src --cov=steps tests/
```

### Code Quality

```bash
# Format code
black src/ steps/ pipelines/

# Lint code
pylint src/ steps/ pipelines/

# Type checking
mypy src/ steps/ pipelines/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

- **MLflow** - Experiment tracking and model deployment
- **ZenML** - Pipeline orchestration and management
- **Scikit-learn** - Machine learning implementations
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation and analysis

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review MLflow/ZenML documentation

---

**Happy Predicting! 🏠📊**
