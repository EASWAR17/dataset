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

## System Architecture 

![architecture diagram](https://github.com/user-attachments/assets/7955d282-e5cb-438b-bfdf-7c7052c70014)


## Output
![image](https://github.com/user-attachments/assets/f418c269-d785-4989-9a11-ab51022fce25)

![image](https://github.com/user-attachments/assets/e6fc2e79-43b6-41a2-a0b1-e6e06418d045)


## Results

The Explainable AI (XAI) for Lung Cancer Detection system successfully integrates deep learning-based medical image analysis with AI-driven interpretability to enhance transparency in model predictions. By leveraging the VGG16 model for lung cancer detection and Grad-CAM for visual explainability, the system highlights critical regions in CT scans that influence its decisions.

The Streamlit-based interactive interface enables seamless user interaction, allowing medical professionals and researchers to upload CT scan images, visualize predictions with heatmap overlays, and receive AI-generated explanations via Gemini API/OpenAI API. This ensures a trustworthy and interpretable AI-powered diagnostic tool, facilitating better decision-making in healthcare. With its modular architecture for model training, inference, and visualization, the system provides a scalable and efficient solution for XAI in medical imaging.

## References

[1]	H. Wedisinghe and T. G. I. Fernando, "Explainable AI for Early Lung Cancer Detection: A Path to Confidence," 2024 4th International Conference on Advanced Research in Computing (ICARC), Belihuloya, Sri Lanka, 2024, pp. 13-18, doi: 10.1109/ICARC61713.2024.10499787.

[2]	R. Harari, A. Al-Taweel, T. Ahram and H. Shokoohi, "Explainable AI and Augmented Reality in Transesophageal Echocardiography (TEE) Imaging," 2024 IEEE International Conference on Artificial Intelligence and eXtended and Virtual Reality (AIxVR), Los Angeles, CA, USA, 2024, pp. 306-309, doi: 10.1109/AIxVR59861.2024.00051.

[3]	S. S, A. S, S. S. N and D. S. S. S, "Role of Explainable AI in Medical Diagnostics and Healthcare: A Pilot Study on Parkinson's Speech Detection," 2024 10th International Conference on Control, Automation and Robotics (ICCAR), Orchard District, Singapore, 2024, pp. 289-294, doi: 10.1109/ICCAR61844.2024.10569414.

[4]	U. Pawar, D. O’Shea, S. Rea and R. O’Reilly, "Explainable AI in Healthcare," 2020 International Conference on Cyber Situational Awareness, Data Analytics and Assessment (CyberSA), Dublin, Ireland, 2020, pp. 1-2, doi: 10.1109/CyberSA49311.2020.9139655.

[5]	S. S. Band, A. Yarahmadi, C.-C. Hsu, M. Biyari, M. Sookhak, R. Ameri, I. Dehzangi, A. T. Chronopoulos, and H.-W. Liang, "Application of explainable artificial intelligence in medical health: A systematic review of interpretability methods," *Informatics in Medicine Unlocked*, vol. 40, p. 101286, 2023. [Online]. doi : 10.1016/j.imu.2023.101286.

[6]	K. M. T. Jawad, A. Verma and F. Amsaad, "Prediction Interpretations of Ensemble Models in Chronic Kidney Disease Using Explainable AI," NAECON 2024 - IEEE National Aerospace and Electronics Conference, Dayton, OH, USA, 2024, pp. 391-397, doi: 10.1109/NAECON61878.2024.10670652.

[7]	S. Amirian et al., "Explainable AI in Orthopedics: Challenges, Opportunities, and Prospects," 2023 Congress in Computer Science, Computer Engineering, & Applied Computing (CSCE), Las Vegas, NV, USA, 2023, pp. 1374-1380, doi: 10.1109/CSCE60160.2023.00230.

[8] 	L. Wang, "Deep Learning Techniques to Diagnose Lung Cancer," Cancers, vol. 14, no. 22, p. 5569, Nov. 2022, doi: 10.3390/cancers14225569.

[9]	S. T. Rikta et al., "XML-GBM Lung: An Explainable Machine Learning-Based Application for the Diagnosis of Lung Cancer," Journal of Pathology Informatics, vol. 14, 2023, doi: 10.1016/j.compbiomed.2024.109547.

[10]	A. Shimazaki et al., "Deep Learning-Based Algorithm for Lung Cancer Detection on Chest Radiographs Using the Segmentation Method," Scientific Reports, vol. 12, no. 1, p. 727, Jan. 2022, doi: 10.1038/s41598-021-04667-w.

[11] 	P. N. Srinivasu, N. Sandhya, R. H. Jhaveri, and R. Raut, "From Blackbox to Explainable AI in Healthcare: Existing Tools and Case Studies," Computational Intelligence and Neuroscience, vol. 2022, p. 8167821, 2022. [Online]. doi: 10.1155/2022/8167821.

[12]	D. Dave, H. Naik, S. Singhal, and P. Patel, "Explainable AI meets Healthcare: A Study on Heart Disease Dataset," arXiv preprint, vol. 2011.03195, 2020. [Online]. doi: 10.48550/arXiv.2011.03195.

[13] 	S. Khedkar, V. Subramanian, G. Shinde, and P. Gandhi, "Explainable AI in Healthcare," 2nd International Conference on Advances in Science & Technology (ICAST), K J Somaiya Institute of Engineering & Information Technology, Mumbai, India, Apr. 2019. [Online].

[14]	H. W. Loh, C. P. Ooi, S. Seoni, P. D. Barua, F. Molinari, and U. R. Acharya, "Application of explainable artificial intelligence for healthcare: A systematic review of the last decade (2011–2022)," Computer Methods and Programs in Biomedicine, vol. 226, p. 107161, 2022. [Online]. doi: 10.1016/j.cmpb.2022.107161.
