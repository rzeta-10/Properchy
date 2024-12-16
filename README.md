# Properchy - House Price Prediction System

A production-grade machine learning system for predicting house prices using MLflow and ZenML. This project implements various design patterns and best practices for building maintainable and scalable ML pipelines.

## Features

- 🏗️ End-to-end ML pipeline implementation
- 📊 Comprehensive data analysis tools
- 🔍 Advanced feature engineering
- 🤖 Model training and evaluation
- 📈 MLflow experiment tracking
- 🚀 Model deployment with MLflow
- ♻️ Continuous deployment pipeline
- 🔄 Real-time inference capabilities

## Architecture

The project follows a modular architecture with clear separation of concerns:

- **Data Analysis**: Includes univariate, bivariate, and multivariate analysis tools
- **Feature Engineering**: Implements various transformation strategies
- **Model Pipeline**: Handles training, evaluation, and deployment
- **Deployment**: Continuous deployment with MLflow integration

## Prerequisites

- Python 3.8+
- MLflow
- ZenML
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/properchy.git
cd properchy
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Pipeline

To run the training pipeline:
```bash
python run_pipeline.py
```

### Deployment Pipeline

To deploy the model:
```bash
python run_deployment.py
```

To stop the deployment service:
```bash
python run_deployment.py --stop-service
```

### Making Predictions

Use the sample prediction script:
```bash
python sample_predict.py
```

## Project Structure

```
properchy/
├── analysis/
│   └── analysis_src/           # Analysis tools
├── pipelines/                  # ML pipeline definitions
├── src/                       # Core source code
├── steps/                     # Pipeline steps
├── data/                      # Data directory
├── requirements.txt           # Project dependencies
└── README.md                 # Project documentation
```

## Key Components

### Data Analysis Tools
- Basic Data Inspection
- Univariate Analysis
- Bivariate Analysis
- Multivariate Analysis
- Missing Values Analysis

### Feature Engineering
- Log Transformation
- Standard Scaling
- Min-Max Scaling
- One-Hot Encoding

### Model Pipeline Steps
- Data Ingestion
- Missing Value Handling
- Feature Engineering
- Outlier Detection
- Model Training
- Model Evaluation
- Model Deployment

## Configuration

The project uses configuration files for various settings:

- `.zen/config.yaml`: ZenML configuration
- `config.yaml`: Model and pipeline configuration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- MLflow for experiment tracking and model deployment
- ZenML for pipeline management
- Scikit-learn for machine learning implementations

