import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Load BERT model and tokenizer
st.title("Toxic Comment Classifier (BERT)")

device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
model.load_state_dict(torch.load("bert_finetuned.pth", map_location="cpu"))
model.to(device)
model.eval()

# Toxicity labels
class_labels = {0: "Non-Toxic", 1: "Toxic", 2: "Severely Toxic", 3: "Obscene", 4: "Threat"}

# User Input
user_input = st.text_area("Enter a comment to analyze:", "I hate you!")

if st.button("Analyze Comment"):
    inputs = tokenizer(user_input, padding=True, truncation=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()

    # Display Results
    st.subheader("Prediction:")
    st.write(f"🔥 **Toxicity Level:** {class_labels.get(prediction, 'Unknown')}")  
    st.write(f"📊 **Confidence Score:** {confidence:.2%}")  
