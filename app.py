import re
import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')

ps = PorterStemmer()
lm = WordNetLemmatizer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Fake News Detector")
input_sms = st.text_area("Enter the News")


def transform_text(data):
    data = data.split(' ')
    data = [re.sub('a-zA-Z0-9',' ', title) for title in data]
    data = [d.lower() for d in data]
    data = [d for d in data if d not in stopwords.words()]
#     data = [ps.stem(d) for d in data]
    data = [lm.lemmatize(d) for d in data]
    data = " ".join(data)
    return data

if st.button('Detect'):
    detected_news = transform_text(input_sms)
    vector_input = tfidf.transform([detected_news])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Not Fake")
    else:
        st.header("Fake")
