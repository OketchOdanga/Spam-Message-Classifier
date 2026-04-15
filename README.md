# Spam-Message-Classifier

---

## How it works (explanation for the user)

1. **On first run** – The app checks for `spam_model.pkl` and `vectorizer.pkl`. If missing, it downloads the SMS dataset, cleans the text, trains a Logistic Regression model, saves it, and displays the test accuracy.

2. **Subsequent runs** – It loads the saved model and vectorizer instantly.

3. **Prediction pipeline**:
   - User types a message.
   - The message is cleaned (same steps used during training).
   - The cleaned text is transformed into TF‑IDF features using the saved vectorizer.
   - The model predicts `spam` (1) or `ham` (0) and outputs the confidence.

---

## Future enhancements

- **Add a slider** to adjust the decision threshold (if you want to control false positives).
- **Show top features** that contributed to the decision (using `model.coef_`).
- **Allow uploading a CSV file** with multiple messages to classify in batch.
