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
from ExtractKeyDetails import extract_structured_data_with_huggingface


# Enhanced email processing functions
def normalize_content(text):
    """Normalize email content for consistent comparison"""
    # Remove quoted text
    text = re.sub(r'(?m)^>.*$', '', text)
    # Remove signatures
    text = re.sub(r'(?m)(^-- $|^--\s*$|^_{10,}$).*$', '', text, flags=re.DOTALL)
    # Remove whitespace and case differences
    text = ' '.join(text.lower().split())
    # Remove common email prefixes
    text = re.sub(r'(?i)^(re:|fw:|fwd:)\s*', '', text)
    return text.strip()

def create_content_fingerprint(text):
    """Create a fingerprint for content comparison"""
    normalized = normalize_content(text)
    return hashlib.md5(normalized.encode()).hexdigest()

# 🔹 Function to Parse Email Thread
def parse_email_thread(email_text):
    """Parse an email thread into individual messages with metadata"""
    messages = []
    separators = [
        (r"^From:\s*(.*)\n^Sent:\s*(.*)\n^To:\s*(.*)\n^Subject:\s*(.*)", "outlook"),
        (r"^On\s(.+)\s*wrote:\s*$", "gmail"),
        (r"^-+\s*Original Message\s*-+$", "generic"),
        (r"^From:\s*(.*)\n^Date:\s*(.*)\n^To:\s*(.*)\n^Subject:\s*(.*)", "alternate"),
        (r"^Le\s(.+)\n^À:\s*(.*)\n^Objet:\s*(.*)", "french")
    ]
    
    remaining_text = email_text
    while True:
        for pattern, style in separators:
            match = re.search(pattern, remaining_text, flags=re.MULTILINE | re.IGNORECASE)
            if match:
                # Extract the current message
                message_part = remaining_text[:match.start()].strip()
                if message_part:
                    messages.append({
                        'content': message_part,
                        'style': style,
                        'match': match.groups()
                    })
                remaining_text = remaining_text[match.end():]
                break
        else:
            if remaining_text.strip():
                messages.append({
                    'content': remaining_text.strip(),
                    'style': 'terminal',
                    'match': None
                })
            break
    
    return messages

def analyze_thread_duplicates(messages):
    """Analyze a thread for duplicate content"""
    analysis = {
        'exact_duplicates': set(),
        'near_duplicates': set(),
        'quoted_duplicates': set(),
        'message_count': len(messages),
        'fingerprints': defaultdict(list)
    }
    
    # Create fingerprints for all messages
    for i, msg in enumerate(messages):
        fp = create_content_fingerprint(msg['content'])
        analysis['fingerprints'][fp].append(i)
    
    # Identify exact duplicates
    for fp, indices in analysis['fingerprints'].items():
        if len(indices) > 1:
            analysis['exact_duplicates'].update(indices)
    
    # Identify near-duplicates using TF-IDF
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS))
    try:
        tfidf_matrix = vectorizer.fit_transform([m['content'] for m in messages])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        for i in range(len(messages)):
            for j in range(i+1, len(messages)):
                if similarity_matrix[i,j] > 0.85:  # Threshold for near-duplicates
                    analysis['near_duplicates'].add((i,j))
    except ValueError:
        pass
    
    # Identify quoted duplicates
    for i in range(len(messages)):
        for j in range(i+1, len(messages)):
            content_i = normalize_content(messages[i]['content'])
            content_j = normalize_content(messages[j]['content'])
            
            if content_i in content_j or content_j in content_i:
                analysis['quoted_duplicates'].add((i,j))
    
    return analysis


def extract_text_from_eml(uploaded_file):
    """Enhanced EML processing with thread analysis"""
    try:
        file_content = uploaded_file.getvalue()
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
            
        msg = BytesParser(policy=policy.default).parsebytes(file_content)
        email_text = ""
        
        # Process email body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))
                
                if content_type == "text/plain" and "attachment" not in content_disposition:
                    email_text += part.get_payload(decode=True).decode('utf-8', errors='replace') + "\n"
                elif content_type == "text/html" and "attachment" not in content_disposition:
                    html_content = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    soup = BeautifulSoup(html_content, "html.parser")
                    email_text += soup.get_text(separator="\n").strip() + "\n"
        else:
            if msg.get_content_type() == "text/plain":
                email_text = msg.get_payload(decode=True).decode('utf-8', errors='replace')
            elif msg.get_content_type() == "text/html":
                html_content = msg.get_payload(decode=True).decode('utf-8', errors='replace')
                soup = BeautifulSoup(html_content, "html.parser")
                email_text = soup.get_text(separator="\n").strip()
        
        # Process attachments
        attachments_text = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get("Content-Disposition"))
                if "attachment" in content_disposition or part.get_filename():
                    file_name = part.get_filename()
                    if file_name:
                        try:
                            file_data = part.get_payload(decode=True)
                            file_stream = BytesIO(file_data)
                            
                            if file_name.lower().endswith(".pdf"):
                                attachments_text += extract_text_from_pdf(file_stream) + "\n"
                            elif file_name.lower().endswith(".docx"):
                                doc = docx.Document(file_stream)
                                attachments_text += "\n".join([para.text for para in doc.paragraphs])
                        except Exception as e:
                            st.error(f"Error processing attachment {file_name}: {str(e)}")
                            continue
        
        # Parse and analyze thread
        thread_messages = parse_email_thread(email_text)
        thread_analysis = analyze_thread_duplicates(thread_messages)
        
        # Store analysis in session state
        st.session_state.thread_analysis[uploaded_file.name] = {
            'messages': thread_messages,
            'analysis': thread_analysis,
            'full_text': email_text + "\n\n" + attachments_text
        }
        
        return email_text + "\n\n" + attachments_text
    
    except Exception as e:
        st.error(f"Error processing EML file {uploaded_file.name}: {str(e)}")
        return ""

# Streamlit UI components
def display_thread_analysis(filename):
    """Display detailed thread analysis"""
    if filename in st.session_state.thread_analysis:
        analysis = st.session_state.thread_analysis[filename]
        messages = analysis['messages']
        stats = analysis['analysis']
        
        with st.expander(f"Thread Analysis: {filename}"):
            st.write(f"**Message Count:** {stats['message_count']}")
            st.write(f"**Exact Duplicates:** {len(stats['exact_duplicates'])} messages")
            st.write(f"**Near Duplicates:** {len(stats['near_duplicates'])} pairs")
            st.write(f"**Quoted Duplicates:** {len(stats['quoted_duplicates'])} pairs")
            
            # Display message details in tabs
            tabs = st.tabs([f"Message {i+1}" for i in range(len(messages))])
            for i, tab in enumerate(tabs):
                with tab:
                    msg = messages[i]
                    col1, col2 = st.columns([3,1])
                    
                    with col1:
                        st.text_area("Content", value=msg['content'], height=200, 
                                    key=f"msg_{i}_{filename}")
                    
                    with col2:
                        st.write("**Metadata**")
                        st.write(f"Style: {msg['style']}")
                        if i in stats['exact_duplicates']:
                            st.error("Exact duplicate")
                        if any(i in pair for pair in stats['near_duplicates']):
                            st.warning("Near duplicate")
                        if any(i in pair for pair in stats['quoted_duplicates']):
                            st.info("Quoted content")    

# Main processing function
def process_files(uploaded_files, selected_fields):
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
            extracted_data = extract_structured_data_with_huggingface(preprocessed_text, selected_fields)
            
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
    
    return results, file_texts, file_names    

# 🔹 Function to Detect Duplicate Files
def detect_duplicates(file_texts, file_names):
    """Detects duplicate emails using text similarity (cosine similarity)."""
    if not file_texts:
        st.error("No meaningful text extracted from any of the files.")
        return [], {}

    vectorizer = TfidfVectorizer().fit_transform(file_texts)
    similarity_matrix = cosine_similarity(vectorizer)
    
    duplicate_pairs = []
    duplicate_flags = {name: False for name in file_names}

    for i in range(len(file_texts)):
        for j in range(i + 1, len(file_texts)):
            similarity = similarity_matrix[i, j]
            if similarity > 0.85:  # ✅ Threshold for duplicate detection
                duplicate_pairs.append((file_names[i], file_names[j], round(similarity, 2)))
                duplicate_flags[file_names[i]] = True
                duplicate_flags[file_names[j]] = True

    return duplicate_pairs, duplicate_flags                      





