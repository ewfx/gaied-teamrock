import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
import sys
import os
from io import BytesIO

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from app.ExtractContent import extract_text_from_pdf, extract_text_from_eml, preprocess_text
from app.EmailClassifier import classify_email_with_huggingface
from app.ExtractKeyDetails import extract_structured_data_with_huggingface, detect_duplicates

class TestStreamlitUI(unittest.TestCase):

    @patch('streamlit.file_uploader')
    @patch('streamlit.multiselect')
    @patch('streamlit.text_input')
    @patch('streamlit.progress')
    @patch('streamlit.spinner')
    @patch('streamlit.warning')
    @patch('streamlit.success')
    @patch('streamlit.info')
    @patch('streamlit.markdown')
    @patch('streamlit.dataframe')
    def test_file_processing(self, mock_dataframe, mock_markdown, mock_info, mock_success, mock_warning, mock_spinner, mock_progress, mock_text_input, mock_multiselect, mock_file_uploader):
        # Mock user inputs
        mock_text_input.return_value = "Field1"
        mock_multiselect.return_value = ["Field1"]
        mock_file_uploader.return_value = [MagicMock(name="file1.pdf"), MagicMock(name="file2.eml")]

        # Mock the progress bar
        mock_progress.return_value = MagicMock()

        # Mock the text extraction functions
        with patch('app.ExtractContent.extract_text_from_pdf', return_value="Sample text from PDF"):
            with patch('app.ExtractContent.extract_text_from_eml', return_value="Sample text from EML"):
                with patch('app.ExtractContent.preprocess_text', return_value="Preprocessed text"):
                    with patch('app.EmailClassifier.classify_email_with_huggingface', return_value=("Request Type", 0.95)):
                        with patch('app.ExtractKeyDetails.extract_structured_data_with_huggingface', return_value={"Field1": "Value1"}):
                            with patch('app.ExtractKeyDetails.detect_duplicates', return_value=([], {})):
                                # Run the Streamlit script
                                exec(open(os.path.join(os.path.dirname(__file__), '../../src/app/StreamlitUI.py')).read(), globals())

                                # Check if the results are displayed correctly
                                mock_markdown.assert_any_call(
                                    """
                                    <div class="stSubheader">
                                        Processed Results
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                mock_dataframe.assert_called_once_with(pd.DataFrame([{
                                    "File Name": "file1.pdf",
                                    "Request Type": "Request Type",
                                    "Confidence Score": 0.95,
                                    "Duplicate": False,
                                    "Field1": "Value1"
                                }, {
                                    "File Name": "file2.eml",
                                    "Request Type": "Request Type",
                                    "Confidence Score": 0.95,
                                    "Duplicate": False,
                                    "Field1": "Value1"
                                }]), use_container_width=True)
                                mock_success.assert_called_once_with("All files processed successfully!")

    @patch('streamlit.file_uploader')
    @patch('streamlit.multiselect')
    @patch('streamlit.text_input')
    @patch('streamlit.info')
    def test_no_files_uploaded(self, mock_info, mock_text_input, mock_multiselect, mock_file_uploader):
        # Mock user inputs
        mock_text_input.return_value = "Field1"
        mock_multiselect.return_value = ["Field1"]
        mock_file_uploader.return_value = []

        # Run the Streamlit script
        exec(open(os.path.join(os.path.dirname(__file__), '../../src/app/StreamlitUI.py')).read(), globals())

        # Check if the info message is displayed
        mock_info.assert_called_once_with("Please upload files and select fields to extract in the sidebar to get started.")

if __name__ == '__main__':
    unittest.main()