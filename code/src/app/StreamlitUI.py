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
from EmailClassifier import classify_email_with_huggingface
from ExtractContent import extract_text_from_pdf, preprocess_text
from ExtractKeyDetails import extract_structured_data_with_huggingface
from IdentifyDuplicates  import detect_duplicates, extract_text_from_eml,display_thread_analysis

# 🔹 Ensure this is the first Streamlit command
#st.set_page_config(page_title="📩 Gen AI Orchestrator for Email and Document Triage/Routing", layout="wide")
# 🔹 Streamlit UI with Enhanced Look and Feel
st.set_page_config(page_title="Gen AI Orchestrator", layout="wide")


# 🔹 Custom CSS for Styling
st.markdown(
    """
    <style>
    /* General App Styling */
    .stApp {
        background-color: #ffffff;  /* White background */
    }
    .stSidebar {
        background-color: #ffffff;  /* White sidebar */
        border-right: 1px solid #e0e0e0;  /* Subtle border */
    }
    .stHeader {
        color: #ffffff;  /* White header text */
        font-size: 2rem;
        font-weight: bold;
        padding: 10px;
        background-color: #006D77;  /* Dark teal header background */
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stSubheader {
        color: #2c3e50;  /* Dark blue subheader text */
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }

    /* Sidebar Button Styling */
    .stButton button {
        background-color: #006D77;  /* Dark teal button */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }

    /* Multiselect Dropdown Styling */
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #ffffff !important;  /* White background */
        border: 1px solid #e0e0e0 !important;  /* Light gray border */
        border-radius: 5px !important;         /* Rounded corners */
    }
    /* Selected Items in Multiselect */
    .stMultiSelect span[data-baseweb="select"] div[role="button"] {
        background-color: #006D77 !important;  /* Dark teal background */
        color: white !important;               /* White text */
        border-radius: 5px !important;         /* Rounded corners */
    }
    .stMultiSelect div[data-baseweb="select"] div[role="button"]:hover {
        background-color: #005A63 !important;  /* Slightly darker teal on hover */
    }

    /* DataFrame Styling */
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stDataFrame th {
        background-color: #006D77 !important;  /* Dark teal header background */
        color: white !important;               /* White header text */
        font-weight: bold !important;          /* Bold header text */
    }

    /* Alerts and Messages */
    .stSuccess {
        background-color: #d4edda;  /* Light green success background */
        color: #155724;  /* Dark green success text */
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3cd;  /* Light yellow warning background */
        color: #856404;  /* Dark yellow warning text */
        border-radius: 5px;
        padding: 10px;
    }
    .stInfo {
        background-color: #d1ecf1;  /* Light blue info background */
        color: #0c5460;  /* Dark blue info text */
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔹 Main Title with Elegant Header
st.markdown(
    """
    <div class="stHeader">
        Gen AI Orchestrator for Email and Document Triage/Routing
    </div>
    """,
    unsafe_allow_html=True,
)

# 🔹 Sidebar for User Inputs
with st.sidebar:
    st.markdown(
        """
        <div class="stSubheader">
            Settings
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("Configure the fields and upload your files here.")

    # 🔹 Free-Text Key Details Entry
    st.subheader("Enter Key Details to Extract")
    new_field = st.text_input("Type a field and press Enter to add it")

    # Store user-entered fields
    if "user_fields" not in st.session_state:
        st.session_state["user_fields"] = []

    # Add new field when user enters and presses Enter
    if new_field and new_field not in st.session_state["user_fields"]:
        st.session_state["user_fields"].append(new_field)

    # Dynamic multi-select dropdown for user-entered fields
    selected_fields = st.multiselect(
        "Select fields to extract from the email content:",
        options=st.session_state["user_fields"],
        default=st.session_state["user_fields"]
    )

    # 🔹 File Uploader
    uploaded_files = st.file_uploader(
        "Upload Multiple Files",
        type=["eml", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload EML, PDF, or DOCX files for processing."
    )

# 🔹 Main Content Area
# 🔹 Main Content Area
if uploaded_files and selected_fields:
    st.markdown("""
        <div class="stSubheader">
            Processing Files...
        </div>
    """, unsafe_allow_html=True)
    
    results = []
    file_texts = []
    file_names = []
    
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        with st.spinner(f"Processing {file.name}..."):
            progress_bar.progress((i + 1) / total_files)
            
            if file.name.lower().endswith(".pdf"):
                extracted_text = extract_text_from_pdf(file)
            else:
                extracted_text = extract_text_from_eml(file)
                
            if not extracted_text.strip():
                st.warning(f"No text extracted from {file.name}. Skipping...")
                continue
                
            preprocessed_text = preprocess_text(extracted_text)
            if not preprocessed_text.strip():
                st.warning(f"No meaningful text extracted from {file.name}. Skipping...")
                continue
                
            file_texts.append(preprocessed_text)
            file_names.append(file.name)
            
            # Classify email intent and extract structured data
            request_type, confidence = classify_email_with_huggingface(preprocessed_text)
            
            try:
                extracted_data = extract_structured_data_with_huggingface(preprocessed_text, selected_fields)
            except Exception as e:
                extracted_data = {field: None for field in selected_fields}
            
            # Check for thread duplicates
            is_duplicate = False
            if file.name in st.session_state.thread_analysis:
                analysis = st.session_state.thread_analysis[file.name]['analysis']
                is_duplicate = len(analysis['exact_duplicates']) > 0 or \
                              len(analysis['near_duplicates']) > 0 or \
                              len(analysis['quoted_duplicates']) > 0
            
            results.append({
                "File Name": file.name,
                "Request Type": request_type,
                "Confidence Score": confidence,
                "Duplicate": is_duplicate,
                **{field: extracted_data.get(field, None) for field in selected_fields}
            })
    
    # Display Processed Results
    st.markdown("""
        <div class="stSubheader">
            Processed Results
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    # Display thread analysis for each EML file
    for file in uploaded_files:
        if file.name.lower().endswith('.eml'):
            display_thread_analysis(file.name)
    
    # Detect duplicates across files
    duplicate_pairs, duplicate_flags = detect_duplicates(file_texts, file_names)
    
    if duplicate_pairs:
        st.markdown("""
            <div class="stSubheader">
                Duplicate Files Detected
            </div>
        """, unsafe_allow_html=True)
        for pair in duplicate_pairs:
            st.warning(f"{pair[0]} and {pair[1]} are duplicates (Similarity: {pair[2]}).")
    
    st.success("All files processed successfully!")
else:
    st.info("Please upload files and select fields to extract in the sidebar to get started.")

