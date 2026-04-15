import streamlit as st
import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Helper functions
def clean_text(text):
    """Basic text cleaning: lower, remove punctuation/digits, extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_and_save_models():
    """Load dataset, train model, save to disk."""
    st.info("Training model for the first time... This may take a few seconds.")
    
    # Load dataset (SMS Spam Collection)
    url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    
    # Clean messages
    df['cleaned'] = df['message'].apply(clean_text)
    
    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['cleaned']).toarray()
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate (optional, for feedback)
    accuracy = model.score(X_test, y_test)
    st.success(f"Training complete! Model accuracy on test set: {accuracy:.2%}")
    
    # Save model and vectorizer
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    return model, vectorizer

def load_models():
    """Load pre-trained model and vectorizer; if missing, train them."""
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return train_and_save_models()

# Streamlit UI 
st.set_page_config(page_title="Spam Message Detector", page_icon="📧")
st.title("📧 Spam Message Classifier")
st.markdown("Enter an SMS or email message below, and the model will tell you if it's **Spam** or **Not Spam**.")

# Load or train models
with st.spinner("Loading model..."):
    model, vectorizer = load_models()

# Text input
user_input = st.text_area("Message:", height=150, placeholder="Type or paste a message here...")

# Predict button
if st.button("Classify", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        cleaned = clean_text(user_input)
        # Transform using the loaded vectorizer
        vec = vectorizer.transform([cleaned])
        # Predict
        prediction = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        
        # Display result
        if prediction == 1:
            st.error(f" **Spam** (confidence: {proba[1]:.2%})")
        else:
            st.success(f" **Not Spam** (confidence: {proba[0]:.2%})")

# Optional: show model info
with st.expander("ℹ️ About this app"):
    st.markdown("""
    - **Dataset**: SMS Spam Collection (5,574 messages)
    - **Preprocessing**: lowercasing, punctuation removal, digit removal
    - **Features**: TF‑IDF (top 5000 words)
    - **Model**: Logistic Regression
    - **Accuracy**: ~98% on test set
    """)