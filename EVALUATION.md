# Model Evaluation Notes

This project includes code paths for evaluating phishing-classification performance with:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive rate

The training workflow uses a stratified **80/20 train/test split** with a fixed random seed for reproducibility.

## Important Portfolio Note

The repository contains the evaluation logic, but this document does **not** claim a specific production-grade accuracy figure unless it is reproduced from an actual run using a defined dataset and environment.

That distinction matters because phishing-detection performance can vary substantially depending on:
- Dataset quality
- Class balance
- Feature distribution
- Model version
- Train/test leakage
- Real-world concept drift

## Recommended Reproducible Evaluation Workflow

1. Use a documented dataset.
2. Record the number of benign and phishing samples.
3. Use the built-in stratified train/test split.
4. Train the models.
5. Capture accuracy, precision, recall and F1.
6. Save the confusion matrix.
7. Document false-positive behaviour.
8. Repeat after feature/model changes.

## Security Interpretation

For phishing detection, recall is important because missed phishing samples can represent undetected threats. Precision is also important because excessive false positives can create analyst fatigue.

A useful SOC-oriented evaluation should therefore discuss the trade-off between:
- detecting as many malicious samples as possible, and
- keeping false positives low enough for practical triage.

## Future Improvements

- Add a reproducible public dataset notebook
- Save evaluation metrics to JSON
- Export a confusion-matrix image
- Add unit tests for feature extraction
- Add GitHub Actions for automated tests
- Add explainability for individual predictions
