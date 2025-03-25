import streamlit as st
import pdfplumber
import docx
import re
import email
import requests
import json
from email import policy
from email.parser import BytesParser
from io import BytesIO
from transformers import pipeline
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# 🔹 Hugging Face API Details
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_API_TOKEN = "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"  # 🔹 Replace with your token

# 🔹 Function to Call Hugging Face API for Email Intent Classification
def classify_email_with_huggingface(text):
    """Sends email content to Hugging Face API for intent classification."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    prompt = f"""
Please classify the following email content into one of the predefined request
types: Adjustment, AU Transfer, Closing Notice, Commitment Change, Fee Payment, Money Movement Inbound, Money Movement Outbound.
The email content is as follows: Email Content: {text}.
After thoroughly analyzing the content, return the result in the following format:
Intent: <intent>, Confidence: <confidence>.
Make sure to focus solely on these parameters and avoid including any additional information or
parameters in your response.
    """

    payload = {"inputs": prompt}
    response = requests.post(HUGGINGFACE_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                generated_text = result[0]["generated_text"]
                input_prompt_end = "parameters in your response."
                if input_prompt_end in generated_text:
                    generated_text = generated_text.split(input_prompt_end)[-1].strip()
                
                intent_match = re.search(r'(Intent|Intention|Category):\s*([^\n,]+)', generated_text, re.IGNORECASE)
                intent = intent_match.group(2).strip() if intent_match else "Unknown"
                
                confidence_match = re.search(r'(Confidence):\s*([0-9.%]+|High|Medium|Low)', generated_text, re.IGNORECASE)
                confidence_text = confidence_match.group(2).strip() if confidence_match else "0.0"
                
                if "%" in confidence_text:
                    confidence = float(confidence_text.replace("%", "")) / 100
                elif confidence_text.lower() in ["high", "medium", "low"]:
                    confidence_mapping = {
                        "high": 0.9,
                        "medium": 0.6,
                        "low": 0.3
                    }
                    confidence = confidence_mapping.get(confidence_text.lower(), 0.0)
                else:
                    confidence = float(confidence_text)
                
                return intent, confidence
            else:
                st.error("⚠️ API returned unexpected format.")
                return "Unknown", 0.0
        except Exception as e:
            st.error(f"Error parsing Hugging Face API response: {e}")
            return "Unknown", 0.0
    else:
        st.error(f"API Request Failed: {response.status_code} - {response.text}")
        return "Unknown", 0.0


