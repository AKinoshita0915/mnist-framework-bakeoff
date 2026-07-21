# End-to-End MNIST Classification: Framework Bake-off

## Overview
This repository demonstrates an approach to building, evaluating, and comparing AI models. Rather than relying on a single algorithm, this project builds and evaluates an image classification pipeline across three distinct frameworks: **Scikit-Learn, PyTorch, and TensorFlow**. 

The goal is to demonstrate framework agility, architectural trade-offs, and software engineering best practices for machine learning.

## Model Architectures & Analysis

I engineered three different solutions to compare classical machine learning bounds against deep learning spatial processing:

| Framework | Architecture | Trade-off Analysis |
| :--- | :--- | :--- |
| **Scikit-Learn** | Random Forest Classifier | Serves as a baseline tree-based model using 100 estimators. |
| **Scikit-Learn** | Soft-Voting Ensemble (MLP + K-NN + Random Forest) | High memory footprint at inference; proves classical limits on spatial data. |
| **PyTorch** | Convolutional Neural Network (CNN) | Highly pythonic; custom training loops allow granular gradient control. |
| **TensorFlow** | Convolutional Neural Network (CNN) | Fast `.fit()` API; highly optimized for eventual TFLite edge deployment. |

## Quick Start

To run this project locally, ensure you have Python 3.9+ installed.

1. **Clone the repository:**
   `git clone https://github.com/AKinoshita0915/mnist-framework-bakeoff.git`
   `cd mnist-framework-bakeoff`

2. **Create a virtual environment:**
   `python -m venv venv`
   `source venv/bin/activate`  # On Windows use `venv\Scripts\activate`

3. **Install dependencies:**
   `pip install -r requirements.txt`

4. **Train and Evaluate Models:**
   `python src/models/pt_cnn.py`
   `python src/models/sklearn_ensemble.py`

## Results and Insights
* **Test Accuracies:** The PyTorch CNN achieved the highest accuracy at 99.08%, followed by the TensorFlow CNN at 98.98%, the Scikit-Learn Ensemble at 98.33%, and the Scikit-Learn Random Forest at 96.75%.
* **Spatial Processing Wins:** The CNN architectures vastly outperformed the flattened classical approaches because convolutional layers maintain the 2D spatial hierarchy of the handwritten digits.
* **Inference Latency:** While the Scikit-Learn K-NN model required zero training time, its $O(N \times D)$ inference latency makes it unsuitable for real-time production APIs compared to the $O(1)$ inference of the compiled CNNs.
* **Cloud Model Hosting:** The exported weights for all four models (`sklearn_rf.joblib`, `sklearn_ensemble.joblib`, `pytorch_cnn.pth`, and `tensorflow_cnn.keras`) have been successfully uploaded and hosted on the Hugging Face Hub under the repository `akinoshi/mnist-framework-backoff`.