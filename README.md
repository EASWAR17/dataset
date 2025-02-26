## OBJECTIVE
To develop an Explainable AI (XAI) for healthcare decision support, focusing on lung cancer detection using CT scans. The project integrates CNN-based detection (VGG16), XAI techniques (Grad-CAM), and AI-generated explanations to provide interpretable insights into model predictions.

## ABOUT
This project aims to build an AI-powered decision support system that predicts lung cancer from CT scan images while ensuring transparency and interpretability using Explainable AI (XAI) techniques. The model employs VGG16, a well-known Convolutional Neural Network (CNN), for lung cancer detection. The Grad-CAM method is used to generate heatmaps that highlight critical areas influencing model predictions.

To enhance explainability, Gemini API or OpenAI API is used to generate human-readable justifications from XAI results, making AI insights more accessible to medical professionals.

The system is deployed using Streamlit, providing an interactive dashboard where users can upload CT scan images, view model predictions, and receive AI-generated explanations.

## FEATURES

1. Lung Cancer Detection using CNN (VGG16)
Uses pretrained VGG16 for feature extraction and classification.
Processes CT scan images to classify potential cancerous regions.
2. Explainable AI (XAI) with Grad-CAM
Highlights critical areas influencing model predictions.
Generates heatmaps for visualizing important regions in CT scans.
3. AI-Generated Explanation of Model Predictions
Uses Gemini API / OpenAI API to convert Grad-CAM insights into human-readable explanations.
Bridges the gap between AI predictions and clinical understanding.
4. Interactive Streamlit Dashboard
Web-based UI for uploading CT scans and viewing results.
Displays original images, Grad-CAM heatmaps, and AI-generated justifications.
5. Performance Metrics & Model Evaluation
Computes accuracy, precision, recall, and F1-score.
Generates ROC curves and confusion matrices for assessment.
6. Deployment with Streamlit
No need for a separate backend (Flask/FastAPI).
Single-click deployment using Streamlit Share or Azure App Service.

## REQUIREMENTS

### Hardware Requirements:
- Processor: Intel Core i7 (or equivalent)
- GPU: NVIDIA RTX 3060 (or higher, recommended for deep learning)
- RAM: 16 GB (minimum)
- Storage: 512 GB SSD (minimum)
### Software Requirements:
Operating System:
- Windows 10/11, macOS, or Linux (Ubuntu 20.04 or later)
### Development Tools:
- Programming Language: Python 3.9+
- Framework: Streamlit (for UI and deployment)
### Deep Learning & XAI Libraries:
- TensorFlow/Keras – For training and fine-tuning VGG16
- Grad-CAM – For generating explainability heatmaps
- NumPy & OpenCV – For image processing
### AI-Powered Explanation Services:
- Gemini API  – For AI-driven justifications of model decisions
### Visualization & Metrics:
- Matplotlib & Seaborn – For performance evaluation graphs
- Scikit-learn – For calculating model metrics
