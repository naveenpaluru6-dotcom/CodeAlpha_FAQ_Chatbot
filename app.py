
import streamlit as st
import json
import nltk
import re

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="College FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.main-title {
    text-align: center;
    color: #244a83;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

# Load FAQ dataset
with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

stop_words = set(stopwords.words("english"))

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Prepare FAQ questions
questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

# Find the best matching answer
def get_answer(user_question):
    processed_query = preprocess(user_question)

    if not processed_query:
        return "Please enter a meaningful question."

    query_vector = vectorizer.transform([processed_query])

    similarity = cosine_similarity(query_vector, faq_vectors)

    best_match = similarity.argmax()
    best_score = similarity[0][best_match]

    if best_score < 0.15:
        return "Sorry, I couldn't understand your question. Please try another question."

    return faqs[best_match]["answer"]

# Application heading
st.markdown(
    '<div class="main-title">🤖 College FAQ Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your smart assistant for college-related questions</div>',
    unsafe_allow_html=True
)

st.divider()

st.info("👋 Hello! Welcome to our College FAQ Chatbot. How can I help you today?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Suggested questions
st.subheader("💡 Suggested Questions")

columns = st.columns(2)

for index, faq in enumerate(faqs):
    with columns[index % 2]:
        if st.button(faq["question"], key=f"faq_{index}"):
            st.session_state.pending_question = faq["question"]
            st.rerun()

st.divider()

# Display chat history
st.subheader("💬 Chat with us")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type your question here...")

# Handle suggested question or typed question
question = st.session_state.pop("pending_question", None)

if user_input:
    question = user_input

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    answer = get_answer(question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

# Footer
st.divider()

st.markdown(
    "<center>Developed as part of CodeAlpha AI Internship | Task 2</center>",
    unsafe_allow_html=True
)
