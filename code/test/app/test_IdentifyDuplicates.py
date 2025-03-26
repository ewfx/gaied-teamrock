import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from app.IdentifyDuplicates import extract_text_from_eml
from app.ExtractContent import extract_text_from_pdf, preprocess_text
from app.IdentifyDuplicates import detect_duplicates
from app.EmailClassifier import classify_email_with_huggingface

class TestIdentifyDuplicates(unittest.TestCase):

    @patch('app.IdentifyDuplicates.st')
    @patch('app.IdentifyDuplicates.BytesParser')
    @patch('app.IdentifyDuplicates.BeautifulSoup')
    @patch('app.IdentifyDuplicates.extract_text_from_pdf')
    def test_extract_text_from_eml_plain_text(self, mock_extract_text_from_pdf, mock_BeautifulSoup, mock_BytesParser, mock_st):
        # Mock the uploaded file
        mock_file = MagicMock()
        mock_file.getvalue.return_value = b"plain text email content"

        # Mock the email message
        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_payload.return_value = b"plain text email content"
        mock_BytesParser.return_value.parsebytes.return_value = mock_msg

        # Call the function
        result = extract_text_from_eml(mock_file)

        # Assertions
        self.assertIn("plain text email content", result)
        mock_st.error.assert_not_called()

    @patch('app.IdentifyDuplicates.st')
    @patch('app.IdentifyDuplicates.BytesParser')
    @patch('app.IdentifyDuplicates.BeautifulSoup')
    @patch('app.IdentifyDuplicates.extract_text_from_pdf')
    def test_extract_text_from_eml_html_content(self, mock_extract_text_from_pdf, mock_BeautifulSoup, mock_BytesParser, mock_st):
        # Mock the uploaded file
        mock_file = MagicMock()
        mock_file.getvalue.return_value = b"<html><body>HTML email content</body></html>"

        # Mock the email message
        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/html"
        mock_msg.get_payload.return_value = b"<html><body>HTML email content</body></html>"
        mock_BytesParser.return_value.parsebytes.return_value = mock_msg

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "HTML email content"
        mock_BeautifulSoup.return_value = mock_soup

        # Call the function
        result = extract_text_from_eml(mock_file)

        # Assertions
        self.assertIn("HTML email content", result)
        mock_st.error.assert_not_called()

    @patch('app.IdentifyDuplicates.st')
    @patch('app.IdentifyDuplicates.BytesParser')
    @patch('app.IdentifyDuplicates.BeautifulSoup')
    @patch('app.IdentifyDuplicates.extract_text_from_pdf')
    def test_extract_text_from_eml_with_attachments(self, mock_extract_text_from_pdf, mock_BeautifulSoup, mock_BytesParser, mock_st):
        # Mock the uploaded file
        mock_file = MagicMock()
        mock_file.getvalue.return_value = b"email with attachment"

        # Mock the email message
        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = True
        mock_part = MagicMock()
        mock_part.get_content_type.return_value = "application/pdf"
        mock_part.get_filename.return_value = "test.pdf"
        mock_part.get_payload.return_value = b"%PDF-1.4"
        mock_msg.walk.return_value = [mock_part]
        mock_BytesParser.return_value.parsebytes.return_value = mock_msg

        # Mock PDF extraction
        mock_extract_text_from_pdf.return_value = "Extracted PDF content"

        # Call the function
        result = extract_text_from_eml(mock_file)

        # Assertions
        self.assertIn("Extracted PDF content", result)
        mock_st.error.assert_not_called()

    @patch('app.IdentifyDuplicates.st')
    @patch('app.IdentifyDuplicates.BytesParser')
    @patch('app.IdentifyDuplicates.BeautifulSoup')
    @patch('app.IdentifyDuplicates.extract_text_from_pdf')
    def test_extract_text_from_eml_error_handling(self, mock_extract_text_from_pdf, mock_BeautifulSoup, mock_BytesParser, mock_st):
        # Mock the uploaded file
        mock_file = MagicMock()
        mock_file.getvalue.return_value = b"email content"

        # Mock the email message to raise an exception
        mock_BytesParser.return_value.parsebytes.side_effect = Exception("Parsing error")

        # Call the function
        result = extract_text_from_eml(mock_file)

        # Assertions
        self.assertEqual(result, "")
        mock_st.error.assert_called_once_with("Error processing EML file <MagicMock name='mock.name' id='...'>: Parsing error")

if __name__ == '__main__':
    unittest.main()