# Medical-Image-classification-using-MLflow-DVC
# Kidney Disease Classification with MLflow & DVC

This project is an end-to-end deep learning pipeline for classifying kidney diseases from CT scan images. It leverages state-of-the-art deep learning models, MLflow for comprehensive experiment tracking, and DVC for data versioning and pipeline reproducibility. A Flask-based REST API provides real-time model inference, while containerization and CI/CD pipelines enable scalable production deployment on cloud platforms.

---

## Project Overview

- **Deep Learning Pipeline:**  
  A CNN-based model classifies kidney diseases (Normal, Cyst, Tumor, Stone) from CT scans. The pipeline is organized into multiple stages:
  - **Data Ingestion:** Downloads the dataset from a remote URL (via Google Drive), extracts the compressed data, and prepares it for training.
  - **Prepare Base Model:** Loads a pre-trained VGG16 model (with ImageNet weights), adapts it for transfer learning by freezing layers and adding custom classification layers.
  - **Model Training:** Trains the updated model using data augmentation and image generators, splitting the dataset into training and validation subsets.
  - **Model Evaluation:** Evaluates the trained model on the validation set, logs evaluation metrics (loss, accuracy) via MLflow, and saves scores in JSON format.
  - **Inference:** A prediction pipeline loads the trained model and, via a Flask REST API, processes base64-encoded images to output classification results (e.g., "Tumor" or "Normal").

- **Experiment Tracking & Reproducibility:**  
  - **MLflow:** Logs parameters, metrics, and artifacts to enable comprehensive experiment tracking.
  - **DVC:** Versions data and orchestrates the entire pipeline (with stages defined in `dvc.yaml`), ensuring full reproducibility and efficient collaboration.

- **REST API:**  
  A Flask API (with Flask-CORS) is implemented for both training (triggered via DVC) and real-time model inference.

- **Utility & Configuration Management:**  
  Utility functions in `common.py` support YAML/JSON read/write operations, directory management, and image encoding/decoding. The `ConfigurationManager` (in `configuration.py`) reads YAML config files and returns structured configuration entities for each pipeline stage.

- **Deployment & CI/CD:**  
  Docker and AWS-based CI/CD workflows are configured (using GitHub Actions) to containerize and deploy the application on a scalable cloud environment.

---

## Technologies and Tools

- **Deep Learning Framework:** TensorFlow 2.12, VGG16 (pre-trained on ImageNet)
- **Experiment Tracking:** MLflow 2.2.2
- **Data Versioning & Pipeline Orchestration:** DVC
- **Web Framework:** Flask, Flask-CORS, Jinja2
- **Data Processing & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Containerization & Deployment:** Docker, AWS (EC2, ECR), GitHub Actions
- **Scripting & Configuration:** Python, Conda, YAML, python-dotenv, Box (ConfigBox)
- **Utilities:** Joblib, ensure, PyYAML, gdown, python-docx.

---


Additional files and tools include:
- **template.py:** Automates project scaffold creation.
- **Dockerfile & Deployment Scripts:** For containerization and cloud deployment.
- **CI/CD Configurations:** GitHub Actions workflows for automated testing and deployment.
- **DVC Lock Files & DAG Visualizations:** Managed by DVC to track pipeline dependencies.

---

## Installation and Setup

### Prerequisites

- **Python 3.8+**
- **Azure Cosmos DB:** Set up an account and obtain your database URL, key, and database name.
- **Environment Variables:**  
  Create a `.env` file in the project root with:
  ```ini
  SECRET_KEY=your_secret_key


  ```markdown
## Setup and Execution Instructions

### Conda Environment Setup
- Create and activate a Conda environment with Python 3.8:
  ```bash
  conda create -n cnncls python=3.8 -y
  conda activate cnncls
  ```

### Install Dependencies
- Install all required Python packages from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### MLflow & DVC Setup
- **Initialize DVC (if not already initialized):**
  ```bash
  dvc init
  ```
- **Reproduce the Pipeline:**
  ```bash
  dvc repro
  ```
- **Launch MLflow UI to Monitor Experiments:**
  ```bash
  mlflow ui
  ```

### Running the Application

#### Pipeline Execution and Training
- **Full Pipeline Execution:**  
  Run the orchestrator script to execute all stages sequentially:
  ```bash
  python main.py
  ```
  This script sequentially executes:
  - **Stage 01:** Data Ingestion – Downloads and extracts the CT scan dataset.
  - **Stage 02:** Prepare Base Model – Loads a pre-trained VGG16 model and updates it for transfer learning.
  - **Stage 03:** Model Training – Trains the updated model using augmented image generators.
  - **Stage 04:** Model Evaluation – Evaluates the trained model and logs metrics via MLflow.

- **Trigger Training via Flask API:**  
  Alternatively, start the Flask server and trigger training through the API:
  ```bash
  python app.py
  ```
  Then use the `/train` endpoint (via your browser or API client) to trigger the DVC pipeline and training process.

#### Inference and Prediction
- **Run the Flask API for Inference:**  
  The API (running on port 8080) exposes the `/predict` endpoint.  
  It accepts a POST request with a base64-encoded image, decodes the image, runs the prediction pipeline (see `prediction.py`), and returns the classification result (e.g., "Tumor" or "Normal").

### Customization
- **Configuration Files:**  
  Modify `config/config.yaml` and `params.yaml` to update project settings such as data paths, image size, batch size, and number of epochs.
  
- **Pipeline and Model Settings:**  
  Adjust the configuration entities in `src/cnnClassifier/entity/config_entity.py` and the `ConfigurationManager` in `src/cnnClassifier/config/configuration.py` to fine-tune each stage of the pipeline.
  
- **Utility Functions:**  
  Extend or customize the utility functions in `common.py` (located at both the project root and in `src/cnnClassifier/utils/`) for tasks like custom logging, file management, and image processing.
  
- **Deployment:**  
  Update the Dockerfile, CI/CD workflows (e.g., GitHub Actions), and AWS deployment scripts to suit your specific environment requirements.
```




