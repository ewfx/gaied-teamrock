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


# 🔹 Function to Extract Text from PDF
def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file using pdfplumber."""
    try:
        with pdfplumber.open(file_stream) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# # 🔹 Function to Extract Text from Email & Attachments
# def extract_text_from_eml(uploaded_file):
#     """Extracts text from uploaded EML file (email body + attachments)."""
#     msg = BytesParser(policy=policy.default).parsebytes(uploaded_file.getvalue())

#     email_text = ""

#     # Extract plain text if available
#     if msg.get_body(preferencelist=("plain",)):
#         email_text = msg.get_body(preferencelist=("plain",)).get_content()
    
#     # Extract from HTML if plain text is missing
#     elif msg.get_body(preferencelist=("html",)):
#         html_content = msg.get_body(preferencelist=("html",)).get_content()
#         soup = BeautifulSoup(html_content, "html.parser")
#         email_text = soup.get_text(separator="\n").strip()

#     attachments_text = ""

#     # Process Attachments
#     for part in msg.iter_attachments():
#         file_name = part.get_filename()
#         if file_name:
#             file_data = part.get_payload(decode=True)
#             file_stream = BytesIO(file_data)

#             # Extract text if it's a PDF
#             if file_name.lower().endswith(".pdf"):
#                 attachments_text += extract_text_from_pdf(file_stream) + "\n"

#             # Extract text if it's a DOCX
#             elif file_name.lower().endswith(".docx"):
#                 doc = docx.Document(file_stream)
#                 attachments_text += "\n".join([para.text for para in doc.paragraphs])

#     # Combine email body + attachment text
#     full_text = email_text + "\n\n" + attachments_text
#     return full_text.strip()

# 🔹 Function to Preprocess Text
def preprocess_text(text):
    """Preprocesses text by removing stop words and non-alphabetic characters."""
    words = [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(words)
