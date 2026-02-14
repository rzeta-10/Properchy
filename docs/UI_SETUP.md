# UI Setup Guide for Properchy

This guide provides detailed instructions for setting up and accessing the user interfaces for experiment tracking and pipeline monitoring.

## 📊 MLflow UI

MLflow provides a comprehensive web interface for tracking experiments, comparing model runs, and managing the model lifecycle.

### Prerequisites

MLflow is already installed as part of the project dependencies. Verify installation:

```bash
mlflow --version
```

### Starting MLflow UI

**Option 1: Basic Setup (Recommended)**
```bash
# From project root directory
mlflow ui --backend-store-uri file:./mlruns
```

**Option 2: Custom Port**
```bash
# If port 5000 is already in use
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

**Option 3: Custom Host (for remote access)**
```bash
# Allow access from other machines
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0
```

### Accessing MLflow UI

1. **Open your browser**
2. **Navigate to:** `http://localhost:5000`
3. **You should see the MLflow Tracking UI**

### MLflow UI Features

#### 1. Experiments View
- **Location:** Main dashboard
- **Features:**
  - List of all experiments
  - Run counts per experiment
  - Creation timestamps
  - Experiment descriptions

#### 2. Runs Comparison
- **How to access:** Click on experiment name → Select multiple runs → Click "Compare"
- **Features:**
  - Side-by-side parameter comparison
  - Metric visualization
  - Parallel coordinates plot
  - Scatter plot matrix

#### 3. Run Details
- **How to access:** Click on any run ID
- **Sections:**
  - **Parameters:** Model hyperparameters
  - **Metrics:** Performance metrics (MSE, R²)
  - **Artifacts:** Saved models and plots
  - **Tags:** Metadata and labels
  - **Notes:** Run descriptions

#### 4. Model Registry
- **Location:** Top navigation → "Models"
- **Features:**
  - Registered model versions
  - Model staging (None, Staging, Production, Archived)
  - Version comparison
  - Model lineage

### Common MLflow UI Tasks

#### View Latest Run
```
1. Go to http://localhost:5000
2. Click on "my_experiment_name"
3. Runs are sorted by start time (latest first)
4. Click on the latest run to view details
```

#### Compare Model Performance
```
1. Select experiment
2. Check boxes next to runs you want to compare
3. Click "Compare" button
4. View metrics in table or chart format
```

#### Download Model Artifacts
```
1. Open run details
2. Scroll to "Artifacts" section
3. Click on artifact name
4. Click "Download" button
```

---

## 🔄 ZenML Dashboard

ZenML provides a dashboard for monitoring pipeline execution and managing ML workflows.

### Installation

ZenML dashboard requires the server component:

```bash
# Install ZenML with server extras
pip install "zenml[server]"
```

### Starting ZenML Dashboard

**Option 1: Local Server (Default)**
```bash
zenml up
```

**Option 2: Custom Port**
```bash
zenml up --port 8238
```

**Option 3: Background Mode**
```bash
zenml up --blocking=false
```

### Accessing ZenML Dashboard

1. **Start the server:** `zenml up`
2. **Wait for startup message:** "Dashboard running at: http://127.0.0.1:8237"
3. **Open browser:** Navigate to `http://localhost:8237`
4. **Default credentials:**
   - Username: `default`
   - Password: (empty - just press Enter)

### ZenML Dashboard Features

#### 1. Pipeline Runs
- **Location:** Main dashboard
- **Features:**
  - List of all pipeline executions
  - Run status (running, completed, failed)
  - Execution duration
  - Trigger information

#### 2. Pipeline DAG Visualization
- **How to access:** Click on pipeline run → "DAG" tab
- **Features:**
  - Visual representation of pipeline steps
  - Step dependencies
  - Execution status per step
  - Step duration

#### 3. Artifacts Browser
- **How to access:** Pipeline run → "Artifacts" tab
- **Features:**
  - Input/output artifacts for each step
  - Artifact metadata
  - Artifact versioning
  - Download capabilities

#### 4. Stack Management
- **Location:** Top navigation → "Stacks"
- **Features:**
  - View configured stacks
  - Component details (orchestrator, artifact store, etc.)
  - Stack switching
  - Component configuration

### Common ZenML Dashboard Tasks

#### Monitor Pipeline Execution
```
1. Go to http://localhost:8237
2. Click on "Pipelines" in sidebar
3. Select your pipeline (e.g., "ml_pipeline")
4. View run history and status
```

#### Inspect Pipeline Step
```
1. Open pipeline run
2. Click on step name in DAG
3. View:
   - Input artifacts
   - Output artifacts
   - Step configuration
   - Execution logs
```

#### View Model Artifacts
```
1. Navigate to pipeline run
2. Click "Artifacts" tab
3. Find "sklearn_pipeline" artifact
4. View metadata and download options
```

### Stopping ZenML Dashboard

```bash
# Stop the server
zenml down
```

---

## 🔧 Troubleshooting

### MLflow UI Issues

#### Issue: "Port 5000 already in use"
**Solution:**
```bash
# Use a different port
mlflow ui --backend-store-uri file:./mlruns --port 5001
```

#### Issue: "No experiments found"
**Solution:**
```bash
# Verify mlruns directory exists
ls mlruns/

# Run training pipeline to create experiments
python run_pipeline.py
```

#### Issue: "Cannot connect to tracking server"
**Solution:**
```bash
# Check if MLflow UI is running
ps aux | grep mlflow

# Restart MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

### ZenML Dashboard Issues

#### Issue: "zenml: command not found"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify ZenML installation
pip install zenml
```

#### Issue: "Dashboard not loading"
**Solution:**
```bash
# Check server status
zenml status

# Restart server
zenml down
zenml up
```

#### Issue: "No pipelines visible"
**Solution:**
```bash
# Verify ZenML is initialized
zenml init

# Check stack configuration
zenml stack describe

# Run a pipeline
python run_pipeline.py
```

---

## 📱 Mobile Access

### Access from Mobile Device (Same Network)

1. **Find your computer's IP address:**
   ```bash
   # Linux/Mac
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. **Start MLflow with host binding:**
   ```bash
   mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0
   ```

3. **Access from mobile:**
   - Open browser on mobile
   - Navigate to: `http://YOUR_IP:5000`
   - Example: `http://192.168.1.100:5000`

---

## 🎨 UI Customization

### MLflow UI Theme

MLflow UI appearance can be customized through browser extensions or by modifying CSS.

### ZenML Dashboard Theme

ZenML dashboard supports light/dark mode:
- Click user icon in top right
- Select "Settings"
- Choose theme preference

---

## 📊 Best Practices

### For MLflow UI

1. **Organize Experiments:** Use descriptive experiment names
2. **Tag Runs:** Add tags for easy filtering
3. **Document Parameters:** Use clear parameter names
4. **Save Artifacts:** Store model files and visualizations
5. **Use Notes:** Add run descriptions for context

### For ZenML Dashboard

1. **Monitor Regularly:** Check pipeline status frequently
2. **Review Artifacts:** Inspect outputs for quality
3. **Track Lineage:** Use DAG view to understand dependencies
4. **Manage Stacks:** Keep stack configurations organized
5. **Archive Old Runs:** Clean up completed pipelines

---

## 🚀 Advanced Features

### MLflow Model Serving

Serve models directly from MLflow:

```bash
# Serve a specific model version
mlflow models serve -m runs:/<RUN_ID>/model -p 5002

# Test the endpoint
curl -X POST http://localhost:5002/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

### ZenML Remote Deployment

Deploy ZenML server for team collaboration:

```bash
# Deploy to cloud (requires additional setup)
zenml deploy --provider aws
```

---

## 📚 Additional Resources

- **MLflow Documentation:** https://mlflow.org/docs/latest/index.html
- **ZenML Documentation:** https://docs.zenml.io/
- **MLflow UI Guide:** https://mlflow.org/docs/latest/tracking.html#tracking-ui
- **ZenML Dashboard Guide:** https://docs.zenml.io/user-guide/starter-guide/dashboard

---

**Need Help?** Open an issue on GitHub or check the troubleshooting section above.
