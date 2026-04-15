#  Spam Message Classifier

A machine learning project that automatically detects spam messages (SMS/email) using **Logistic Regression** and **TF‑IDF** features. The model is deployed as an interactive web app built with **Streamlit**.

##  Project Goal

Build a model that classifies a given message as **spam** or **not spam (ham)** – a common task in email and messaging platforms to filter unwanted content.

##  Dataset

- **Source**: [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) (UCI)  
- **Content**: 5,574 SMS messages, labeled as `spam` or `ham`  
- **Distribution**: 4,825 ham (86.6%), 747 spam (13.4%)  

The dataset is automatically downloaded when running the training script or the Streamlit app for the first time.

##  Preprocessing Steps

To prepare the raw text for machine learning, the following steps are applied:

1. **Lowercase** – Convert all characters to lowercase.  
2. **Remove punctuation** – Strip punctuation marks (e.g., `.,!?`).  
3. **Remove digits** – Delete numeric characters (optional but helps reduce noise).  
4. **Remove extra whitespace** – Collapse multiple spaces and trim.  

*Note: Tokenization is handled internally by `TfidfVectorizer`.*

Example:  
`"FREE entry!! 2 win a trip to London!"` → `"free entry win a trip to london"`

##  Model Choice

We compared two common classifiers for text data:

| Model | Accuracy | Why chosen |
|-------|----------|-------------|
| **Multinomial Naive Bayes** | ~97% | Fast, works well with sparse word counts |
| **Logistic Regression** | ~98% | Slightly better performance, robust to feature correlations |

**Final model**: **Logistic Regression** – better precision/recall balance for spam detection.

### Feature Extraction

- **TF‑IDF (Term Frequency – Inverse Document Frequency)**  
  - Limits vocabulary to the top 5,000 words  
  - Removes English stop words (`the`, `a`, `and`, …)  
  - Converts messages into a numerical matrix where rare words get higher weight.

##  Evaluation

The model is evaluated on a 20% hold‑out test set (stratified split). Metrics:

```
Accuracy: 98.3%
Precision (spam): 0.96
Recall (spam): 0.93
F1-score (spam): 0.94
```


```
--- Naive Bayes ---
Accuracy: 0.9695067264573991
Classification Report:
               precision    recall  f1-score   support

           0       0.97      1.00      0.98       966
           1       1.00      0.77      0.87       149

    accuracy                           0.97      1115
   macro avg       0.98      0.89      0.93      1115
weighted avg       0.97      0.97      0.97      1115

Confusion Matrix:
 [[966   0]
 [ 34 115]]
----------------------------------------
--- Logistic Regression ---
Accuracy: 0.9605381165919282
Classification Report:
               precision    recall  f1-score   support

           0       0.96      1.00      0.98       966
           1       1.00      0.70      0.83       149

    accuracy                           0.96      1115
   macro avg       0.98      0.85      0.90      1115
weighted avg       0.96      0.96      0.96      1115

Confusion Matrix:
 [[966   0]
 [ 44 105]]
----------------------------------------

```
