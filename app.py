import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import hf_hub_download

# ✅ Load model from Hugging Face
st.sidebar.title("Settings")
MODEL_PATH = hf_hub_download(repo_id="AtharvaNaik/bert-toxicity", filename="bert_finetuned.pth")

# ✅ Load tokenizer and model
st.title("Atharva Naik 124B2B011 : Toxic Comment Classifier (BERT)")

device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Load model weights from Hugging Face
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()

# ✅ Toxicity Labels
class_labels = {
    0: "Non-Toxic 🟢",
    1: "Toxic 🟡",
    2: "Severely Toxic 🟠",
    3: "Obscene 🔴",
    4: "Threat ⚠️"
}

# ✅ User Input
st.subheader("Enter a comment to analyze:")
user_input = st.text_area("Type here...", "I hate you!")

if st.button("Analyze Comment"):
    # Tokenize input
    inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt").to(device)

    # Predict Toxicity
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    # ✅ Display Results
    st.subheader("Prediction:")
    st.write(f"🔍 **Toxicity Level:** {class_labels.get(prediction, 'Unknown')}")  
    st.write(f"📊 **Confidence Score:** {confidence:.2%}")  
