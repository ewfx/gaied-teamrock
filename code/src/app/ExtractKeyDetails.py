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
from email.utils import parsedate_to_datetime
import hashlib
from collections import defaultdict
from email.utils import parsedate_to_datetime
import hashlib
from collections import defaultdict
import logging


# 🔹 Hugging Face API Details
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HUGGINGFACE_API_TOKEN = "token"  # 🔹 Replace with your token

# 🔹 Function to Clean and Parse JSON from Model Response
def clean_and_parse_json(response_text):
    """
    Cleans the response text and parses it into a JSON object while handling errors gracefully.
    """
    try:
        response_text = response_text.strip()
        response_text = response_text.replace("```", "")  # Remove code block formatting if present
        
        # Extract JSON content using regex
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logging.error("No valid JSON object found in the response.")
            return {"error": "No valid JSON detected."}
        
        json_str = json_match.group(0)
        json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes for JSON compatibility
        json_str = json_str.replace("\n", "")  # Remove unnecessary newlines
        
        # Ensure proper JSON formatting
        try:
            extracted_data = json.loads(json_str)
        except json.JSONDecodeError:
            json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)  # Ensure keys are quoted
            extracted_data = json.loads(json_str)
        
        return extracted_data
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error: {e}")
        return {"error": "Failed to parse JSON."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "Unexpected error occurred."}


# 🔹 Function to Call Hugging Face API for Structured Data Extraction
def extract_structured_data_with_huggingface(text, fields):
    """Sends text to Hugging Face API for structured data extraction."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Dynamically construct the prompt based on user-defined fields
    fields_list = "\n- ".join(fields)
    prompt = f"""
Extract the following fields from the text provided below.
Ensure the extraction is precise and matches the exact requirements.
Return the result in JSON format with the specified structure.
Do not include any additional explanations or text outside the JSON object.

Fields to Extract:

- {fields_list}

Text:
{text}

Output Format:

JSON:
{{
    "Field1": "Extracted Value",
    "Field2": "Extracted Value",
    ...
}}

    """

    payload = {"inputs": prompt}
    response = requests.post(HUGGINGFACE_API_URL, json=payload, headers=headers)

    if response.status_code == 200:
        try:
            result = response.json()
        

            if isinstance(result, list) and "generated_text" in result[0]:
                generated_text = result[0]["generated_text"]

                # Identify and remove the input prompt from the generated text
                # The prompt ends with the JSON structure example
                prompt_end = "JSON:\n{\n    \"Field1\": \"Extracted Value\",\n    \"Field2\": \"Extracted Value\",\n    ...\n}\n"
                if prompt_end in generated_text:
                    generated_text = generated_text.split(prompt_end)[-1].strip()


                # Clean and parse the JSON from the generated text
                extracted_data = clean_and_parse_json(generated_text)
                return extracted_data
            else:
                st.error("⚠️ API returned unexpected format.")
                return {}
        except Exception as e:
            st.error(f"Error parsing Hugging Face API response: {e}")
            return {}
    else:
        st.error(f"API Request Failed: {response.status_code} - {response.text}")
        return {}


# # 🔹 Function to Detect Duplicate Files
# def detect_duplicates(file_texts, file_names):
#     """Detects duplicate emails using text similarity (cosine similarity)."""
#     if not file_texts:
#         st.error("No meaningful text extracted from any of the files.")
#         return [], {}

#     vectorizer = TfidfVectorizer().fit_transform(file_texts)
#     similarity_matrix = cosine_similarity(vectorizer)
    
#     duplicate_pairs = []
#     duplicate_flags = {name: False for name in file_names}

#     for i in range(len(file_texts)):
#         for j in range(i + 1, len(file_texts)):
#             similarity = similarity_matrix[i, j]
#             if similarity > 0.85:  # ✅ Threshold for duplicate detection
#                 duplicate_pairs.append((file_names[i], file_names[j], round(similarity, 2)))
#                 duplicate_flags[file_names[i]] = True
#                 duplicate_flags[file_names[j]] = True

#     return duplicate_pairs, duplicate_flags        
