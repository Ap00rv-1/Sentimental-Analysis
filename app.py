import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join([i for i in text if not i.isdigit()])
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

# Streamlit UI
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter text to analyze:")
emotion_map = {
    0: 'sadness',
    1: 'anger',
    2: 'love',
    3: 'surprise',
    4: 'fear',
    5: 'joy'
}


if st.button("Predict"):
    if user_input:
        processed_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([processed_text])
        prediction_idx = model.predict(vectorized_text)[0]
        
        # Look up the emotion name using the number
        emotion_name = emotion_map.get(prediction_idx, "Unknown")
        
        st.success(f"Predicted Emotion: {emotion_name}")
    else:
        st.warning("Please enter some text.")