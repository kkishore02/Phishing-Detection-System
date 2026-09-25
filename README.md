# AI-Powered Phishing Detection System

A Python-based cybersecurity project that detects potentially malicious URLs and suspicious email content using feature engineering and multiple machine-learning models.

## Overview

This project explores how machine-learning techniques can support phishing detection by analysing characteristics of URLs and email content. The system extracts security-relevant features and uses an ensemble of models to classify suspicious content.

## Key Features

- URL feature extraction and analysis
- Email-content feature extraction
- Suspicious keyword detection
- Shortened URL detection
- Sender and subject analysis
- Random Forest classification
- Support Vector Machine (SVM) classification
- Neural-network classification when TensorFlow is available
- Model evaluation using accuracy, precision, recall, F1-score and confusion matrices
- Model persistence with Joblib

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Joblib

## How It Works

```text
URL / Email Input
       |
       v
Feature Extraction
       |
       v
Feature Scaling
       |
       v
Random Forest / SVM / Neural Network
       |
       v
Phishing Risk Classification
```

### URL Indicators Analysed

Examples include:

- URL and domain length
- IP-address usage
- Number of subdomains
- Suspicious keywords
- URL-shortener domains
- HTTPS usage
- Path depth
- Suspicious file extensions
- Query parameters

### Email Indicators Analysed

Examples include:

- Urgency language
- Suspicious links
- Excessive capitalisation
- Sender characteristics
- Subject-line urgency
- Money/payment references
- Repeated characters and punctuation patterns

## Installation

Clone the repository:

```bash
git clone https://github.com/kkishore02/Phishing-Detection-System.git
cd Phishing-Detection-System
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python phishing_detector.py
```

## Security Relevance

Phishing remains a common initial-access technique. This project demonstrates how automated feature extraction and classification can be used to support defensive security workflows and phishing triage.

## Skills Demonstrated

- Python security automation
- Phishing analysis
- Feature engineering
- Machine learning for cybersecurity
- Security-focused data analysis
- Model evaluation
- Defensive security thinking

## Future Improvements

- Train on larger real-world phishing datasets
- Add explainability for individual classifications
- Add domain-age and reputation enrichment
- Integrate with a SIEM or SOC workflow
- Add unit tests and CI
- Build a lightweight analyst dashboard

## Disclaimer

This project is intended for defensive cybersecurity research, education and portfolio demonstration. Predictions from a machine-learning model should not be treated as a substitute for professional security analysis.

## Author

**Kishore Bandi**  
Cybersecurity MSc Graduate | Aspiring SOC Analyst
