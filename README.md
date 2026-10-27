# Dry Bean Type Classification

## a. Problem statement

Dry beans of seven registered varieties look similar, so they are hard to separate by eye. This project classifies each bean into one variety using 16 size and shape measurements taken from images. The varieties are BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA.

Five saved classifiers are compared on the same held-out test set. A Streamlit app loads those saved models and scores an uploaded test CSV. The app does not retrain the models.

## b. Dataset description

- Source: UCI Machine Learning Repository, Dry Bean Dataset, id 602
- https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- Rows: 13,611 beans
- Features: 16 numeric measurements, with no missing values
- Target: Class (7 varieties)
- Class counts: DERMASON 3546, SIRA 2636, SEKER 2027, HOROZ 1928, CALI 1630, BARBUNYA 1322, BOMBAY 522
- Split: 80% train and 20% test, stratified by class, random_state 42
- Test file used by the app: test_data.csv

Feature columns: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4.

Logistic Regression, KNN, and SVM use a saved StandardScaler inside the model pipeline. Decision Tree and Gaussian Naive Bayes use the raw measurements.

## c. Github Repository Link

https://github.com/Vinod2300/dry-bean-classification

## d. Models used

Precision, Recall, and F1 use a weighted average because the seven classes are unequal in size. AUC is weighted one-vs-rest.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Support Vector Machine | 0.9221 | 0.9884 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

## e. Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Very strong on every metric. Accuracy is 0.9207, and it has the highest AUC at 0.9934, so it separates the seven varieties very well. |
| Decision Tree | Accuracy is 0.8917. It is usable, but it is behind Logistic Regression, KNN, and SVM on every score. A single tree does not generalize as well on this data. |
| KNN | Accuracy is 0.9166 and AUC is 0.9812. It is close to the top two models. Predictions depend on nearby beans in the scaled feature space. |
| Naive Bayes | Weakest classifier. Accuracy is only 0.7639 and MCC is 0.7154. AUC is still high at 0.9644, so the score ranking is better than the final class decision. Gaussian Naive Bayes assumes the features are independent, and these shape measurements are related to each other. |
| Support Vector Machine | Best accuracy (0.9221), best F1 (0.9222), and best MCC (0.9059). AUC is 0.9884, slightly below Logistic Regression. |
| Overall, Winner for your dataset? | Support Vector Machine. It gives the most correct variety labels on the test set. Logistic Regression is a very close second and is the best model if AUC is the main score. |
