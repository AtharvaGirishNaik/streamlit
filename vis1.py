import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

st.title("GPT vs. BERT: How AI Understands Text")

# Load GPT and BERT models
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")

user_input = st.text_input("Enter a sentence:", "AI is revolutionizing the world.")

if st.button("Process Text"):
    # Tokenization for BERT
    bert_tokens = bert_tokenizer.tokenize(user_input)
    st.write("**BERT Tokenization:**", bert_tokens)

    # Tokenization for GPT
    gpt_tokens = gpt_tokenizer.tokenize(user_input)
    st.write("**GPT Tokenization:**", gpt_tokens)

    # GPT Text Generation
    input_ids = gpt_tokenizer.encode(user_input, return_tensors="pt")
    output = gpt_model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

    st.write("**GPT Generated Text:**", generated_text)
