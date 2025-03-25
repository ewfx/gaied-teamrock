import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from app.ExtractContent import extract_text_from_pdf, extract_text_from_eml, preprocess_text



class TestExtractContent(unittest.TestCase):

    @patch('pdfplumber.open')
    def test_extract_text_from_pdf_valid(self, mock_pdfplumber_open):
        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]
        mock_pdf.pages[0].extract_text.return_value = "Page 1 text"
        mock_pdf.pages[1].extract_text.return_value = "Page 2 text"
        mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

        file_stream = BytesIO(b"dummy pdf content")
        extracted_text = extract_text_from_pdf(file_stream)
        self.assertEqual(extracted_text, "Page 1 text\nPage 2 text")

    @patch('pdfplumber.open')
    def test_extract_text_from_pdf_invalid(self, mock_pdfplumber_open):
        mock_pdfplumber_open.side_effect = Exception("Error opening PDF")
        file_stream = BytesIO(b"dummy pdf content")
        extracted_text = extract_text_from_pdf(file_stream)
        self.assertEqual(extracted_text, "")

    @patch('app.ExtractContent.extract_text_from_pdf')
    @patch('docx.Document')
    def test_extract_text_from_eml_valid(self, mock_docx_Document, mock_extract_text_from_pdf):
        mock_extract_text_from_pdf.return_value = "PDF text"
        mock_doc = MagicMock()
        mock_doc.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]
        mock_docx_Document.return_value = mock_doc

        eml_content = b"dummy eml content"
        uploaded_file = BytesIO(eml_content)
        msg = MagicMock()
        msg.get_body.return_value.get_content.return_value = "Email body text"
        msg.iter_attachments.return_value = [
            MagicMock(get_filename=lambda: "attachment.pdf", get_payload=lambda decode: b"pdf content"),
            MagicMock(get_filename=lambda: "attachment.docx", get_payload=lambda decode: b"docx content")
        ]

        with patch('email.parser.BytesParser.parsebytes', return_value=msg):
            extracted_text = extract_text_from_eml(uploaded_file)
            self.assertIn("Email body text", extracted_text)
            self.assertIn("PDF text", extracted_text)
            self.assertIn("Paragraph 1", extracted_text)
            self.assertIn("Paragraph 2", extracted_text)

    def test_preprocess_text(self):
        text = "This is a sample text with stopwords and non-alphabetic characters 123!"
        preprocessed_text = preprocess_text(text)
        self.assertEqual(preprocessed_text, "sample text stopwords non-alphabetic characters 123!")

if __name__ == '__main__':
    unittest.main()