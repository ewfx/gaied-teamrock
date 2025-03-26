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
from EmailClassifier import classify_email_with_huggingface
from ExtractContent import extract_text_from_pdf, preprocess_text
from ExtractKeyDetails import clean_and_parse_json,extract_structured_data_with_huggingface
from IdentifyDuplicates  import detect_duplicates, extract_text_from_eml,display_thread_analysis

# 🔹 Ensure this is the first Streamlit command
#st.set_page_config(page_title="📩 Gen AI Orchestrator for Email and Document Triage/Routing", layout="wide")
# 🔹 Streamlit UI with Enhanced Look and Feel
st.set_page_config(page_title="Gen AI Orchestrator", layout="wide")

# Initialize session state for request type configurations
if 'request_type_config' not in st.session_state:
    st.session_state.request_type_config = {
        "Adjustment": ["date", "account"],
        "AU Transfer": ["source_account", "destination_account", "transfer_amount", "transfer_date"],
        "Closing Notice": ["deal_name", "closing_date", "contact_person", "final_amount"],
        "Commitment Change": ["date", "old_amount", "new_amount", "change_reason"],
        "Fee Payment": ["invoice_number", "payment_amount", "due_date", "payment_method"],
        "Money Movement Inbound": ["sender", "amount", "receiving_account", "expected_date"],
        "Money Movement Outbound": ["recipient", "amount", "sending_account", "transfer_date"]
    }

# Initialize session state for field descriptions
if 'field_descriptions' not in st.session_state:
    st.session_state.field_descriptions = {
        "deal_name": "Typically starts with 'D-' followed by numbers, or contains client name",
        "adjustment_amount": "Numeric value representing the adjustment amount",
        "reason": "Text description of the reason for adjustment",
        "effective_date": "Date when the adjustment takes effect (format: MM/DD/YYYY or similar)",
        "source_account": "Account number or identifier for the source of transfer",
        "destination_account": "Account number or identifier for the destination of transfer",
        "transfer_amount": "Numeric value being transferred between accounts",
        "closing_date": "Date when the deal will be closed (format: MM/DD/YYYY or similar)",
        "contact_person": "Name of the primary contact for this deal",
        "final_amount": "Final numeric value for the deal closure",
        "old_amount": "Original numeric value before commitment change",
        "new_amount": "New numeric value after commitment change",
        "invoice_number": "Alphanumeric identifier for the invoice",
        "payment_amount": "Numeric value of the payment",
        "due_date": "Date when payment is due (format: MM/DD/YYYY or similar)",
        "payment_method": "Method of payment (e.g., wire transfer, check, ACH)",
        "sender": "Name or identifier of the sender for inbound money movement",
        "receiving_account": "Account number or identifier receiving the funds",
        "expected_date": "Date when funds are expected (format: MM/DD/YYYY or similar)",
        "recipient": "Name or identifier of the recipient for outbound money movement",
        "sending_account": "Account number or identifier sending the funds"
    }


# Initialize session state for thread analysis
if 'thread_analysis' not in st.session_state:
    st.session_state.thread_analysis = {}

# Custom CSS for Styling
st.markdown(
    """
    <style>
    /* General App Styling */
    .stApp {
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    .stHeader {
        color: #ffffff;
        font-size: 2rem;
        font-weight: bold;
        padding: 10px;
        background-color: #006D77;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .stSubheader {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #006D77;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMultiSelect div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 5px !important;
    }
    .stMultiSelect span[data-baseweb="select"] div[role="button"] {
        background-color: #006D77 !important;
        color: white !important;
        border-radius: 5px !important;
    }
    .stMultiSelect div[data-baseweb="select"] div[role="button"]:hover {
        background-color: #005A63 !important;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stDataFrame th {
        background-color: #006D77 !important;
        color: white !important;
        font-weight: bold !important;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 10px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-radius: 5px;
        padding: 10px;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title with Elegant Header
st.markdown(
    """
    <div class="stHeader">
        Gen AI Orchestrator for Email and Document Triage/Routing
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for User Inputs
with st.sidebar:
    st.markdown(
        """
        <div class="stSubheader">
            Settings
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Multiple Files",
        type=["eml", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload EML, PDF, or DOCX files for processing."
    )
    
    # Configuration section for request types and fields
    with st.expander("Configure Request Types & Fields"):
        selected_request_type = st.selectbox(
            "Select Request Type to Configure",
            options=list(st.session_state.request_type_config.keys())
        )
        
        # Get current fields for selected request type
        current_fields = st.session_state.request_type_config[selected_request_type]
        
        # Get all available fields (combining existing and new fields)
        all_fields = list(set(
            list(st.session_state.field_descriptions.keys()) + 
            current_fields + 
            ["Add new field"]
        ))
        
        # Add/remove fields for selected request type
        updated_fields = st.multiselect(
            f"Fields for {selected_request_type}",
            options=all_fields,
            default=[f for f in current_fields if f in all_fields]
        )
        
        # Handle new field addition
        if "Add new field" in updated_fields:
            new_field = st.text_input("Enter new field name")
            if new_field and new_field not in st.session_state.field_descriptions:
                st.session_state.field_descriptions[new_field] = ""
                updated_fields.remove("Add new field")
                updated_fields.append(new_field)
                st.session_state.request_type_config[selected_request_type] = updated_fields
                st.experimental_rerun()
        
        # Update fields for request type
        if set(updated_fields) != set(current_fields) and "Add new field" not in updated_fields:
            st.session_state.request_type_config[selected_request_type] = updated_fields
            st.success(f"Updated fields for {selected_request_type}")
        
        # Field descriptions editor
        st.subheader("Field Descriptions")
        selected_field = st.selectbox(
            "Select Field to Edit Description",
            options=list(st.session_state.field_descriptions.keys())
        )
        new_description = st.text_area(
            "Field Description",
            value=st.session_state.field_descriptions[selected_field],
            help="This description helps the AI understand how to extract this field"
        )
        if new_description != st.session_state.field_descriptions[selected_field]:
            st.session_state.field_descriptions[selected_field] = new_description
            st.success(f"Updated description for {selected_field}")

# Main Content Area - Display Current Configuration
st.markdown("""
    <div class="stSubheader">
        Current Field Configuration by Request Type
    </div>
""", unsafe_allow_html=True)

# Display current configuration
config_df = pd.DataFrame.from_dict(
    st.session_state.request_type_config, 
    orient='index'
).transpose()
st.dataframe(config_df, use_container_width=True)


# 🔹 Main Content Area
if uploaded_files:
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

            # Get relevant fields for this request type
            relevant_fields = st.session_state.request_type_config.get(request_type, [])
            
            
            try:
                extracted_data = extract_structured_data_with_huggingface(preprocessed_text, relevant_fields)
            except Exception as e:
                extracted_data = {field: None for field in relevant_fields}
            
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
                **{field: extracted_data.get(field, None) for field in relevant_fields}
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

