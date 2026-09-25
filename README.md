# AI-Powered Phishing Detection System

A Python-based defensive cybersecurity project that analyses URLs and email content using feature engineering and multiple machine-learning models.

## Overview

This project explores how machine learning can support phishing triage by extracting security-relevant indicators from URLs and email content and combining multiple classifiers to estimate phishing risk.

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
- Stratified 80/20 train/test split for reproducibility
- Model persistence with Joblib

## Technologies

Python • Pandas • NumPy • Scikit-learn • TensorFlow/Keras • Joblib

## Detection Workflow

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
Ensemble Risk Classification
       |
       v
Evaluation & Analyst Review
```

## URL Indicators Analysed

Examples include URL/domain length, IP-address usage, number of subdomains, suspicious keywords, shortened URLs, HTTPS usage, path depth, suspicious file extensions and query parameters.

## Email Indicators Analysed

Examples include urgency language, suspicious links, excessive capitalisation, sender characteristics, subject-line urgency, payment language and repeated punctuation patterns.

## Installation

```bash
git clone https://github.com/kkishore02/Phishing-Detection-System.git
cd Phishing-Detection-System
pip install -r requirements.txt
python phishing_detector.py
```

## Evaluation

The code includes evaluation logic for:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive rate

See [EVALUATION.md](./EVALUATION.md) for the reproducibility and security-analysis approach.

> Specific performance figures should only be reported when reproduced from a documented dataset and run environment.

## Security Relevance

Phishing is a common initial-access technique. This project demonstrates how automated feature extraction and classification can support defensive security workflows, while recognising that model output should assist rather than replace analyst judgement.

## Skills Demonstrated

- Python security automation
- Phishing analysis
- Feature engineering
- Machine learning for cybersecurity
- Security-focused data analysis
- Model evaluation
- Defensive security thinking

## Future Improvements

- Add a reproducible public dataset notebook
- Export evaluation metrics and confusion-matrix images
- Add explainability for predictions
- Add domain-age and reputation enrichment
- Integrate with a SIEM/SOC workflow
- Add unit tests and CI
- Build an analyst dashboard

## Disclaimer

This project is intended for defensive cybersecurity research, education and portfolio demonstration. Predictions from a machine-learning model should not be treated as a substitute for professional security analysis.

## Author

**Kishore Bandi**  
MSc Cyber Security & Penetration Testing | Aspiring SOC / Cyber Security Analyst
