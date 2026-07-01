import streamlit as st
import torch
import torch.nn as nn
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Page Config

st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="📩",
    layout="centered"
)

# Model Definition

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
    
# Load Vectorizer

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

INPUT_SIZE = 5000
model = RNN(INPUT_SIZE)
model.load_state_dict(torch.load("spam_model.pth", map_location="cpu"))
model.eval()

# Preprocessing

ps = PorterStemmer()

def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    stemmed_words = []

    for token in tokens:
        stemmed_words.append(ps.stem(token))

    return " ".join(stemmed_words)

# prediction

def predict(text):

    processed = preprocess(text)

    vector = tfidf.transform([processed])

    vector = vector.toarray()

    tensor = torch.tensor(vector, dtype=torch.float32)

    tensor = tensor.unsqueeze(1)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(output).item()

    prediction = 1 if probability > 0.5 else 0

    return prediction, probability

# streamlit UI

st.title("📩 Spam Message Detector")

st.write(
    "Detect whether a text message is **Spam** or **Ham** using a trained RNN model."
)

message = st.text_area(
    "Enter your message",
    height=180
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:

        prediction, prob = predict(message)

        if prediction == 1:
            st.error("🚨 Spam Message")
            st.progress(prob)
            st.write(f"Confidence : **{prob*100:.2f}%**")

        else:
            st.success("✅ Ham Message")
            st.progress(1-prob)
            st.write(f"Confidence : **{(1-prob)*100:.2f}%**")