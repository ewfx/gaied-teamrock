import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from app.ExtractKeyDetails import clean_and_parse_json, extract_structured_data_with_huggingface, detect_duplicates



class TestExtractKeyDetails(unittest.TestCase):

    def test_clean_and_parse_json_valid(self):
        response_text = """
        ```json
        {
            'Field1': 'Value1',
            'Field2': 'Value2'
        }
        ```
        """
        expected_output = {
            "Field1": "Value1",
            "Field2": "Value2"
        }
        parsed_json = clean_and_parse_json(response_text)
        self.assertEqual(parsed_json, expected_output)

    def test_clean_and_parse_json_invalid(self):
        response_text = "Invalid JSON response"
        parsed_json = clean_and_parse_json(response_text)
        self.assertEqual(parsed_json, {})

    @patch('requests.post')
    def test_extract_structured_data_with_huggingface_valid(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{
            "generated_text": """
            JSON:
            {
                "Field1": "Value1",
                "Field2": "Value2"
            }
            """
        }]
        mock_post.return_value = mock_response

        text = "Sample text"
        fields = ["Field1", "Field2"]
        expected_output = {
            "Field1": "Value1",
            "Field2": "Value2"
        }
        extracted_data = extract_structured_data_with_huggingface(text, fields)
        self.assertEqual(extracted_data, expected_output)

    @patch('requests.post')
    def test_extract_structured_data_with_huggingface_invalid(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        text = "Sample text"
        fields = ["Field1", "Field2"]
        extracted_data = extract_structured_data_with_huggingface(text, fields)
        self.assertEqual(extracted_data, {})

    def test_detect_duplicates_no_texts(self):
        file_texts = []
        file_names = []
        duplicate_pairs, duplicate_flags = detect_duplicates(file_texts, file_names)
        self.assertEqual(duplicate_pairs, [])
        self.assertEqual(duplicate_flags, {})

    def test_detect_duplicates_with_duplicates(self):
        file_texts = ["This is a test.", "This is a test."]
        file_names = ["file1.txt", "file2.txt"]
        duplicate_pairs, duplicate_flags = detect_duplicates(file_texts, file_names)
        self.assertEqual(duplicate_pairs, [("file1.txt", "file2.txt", 1.0)])
        self.assertEqual(duplicate_flags, {"file1.txt": True, "file2.txt": True})

    def test_detect_duplicates_no_duplicates(self):
        file_texts = ["This is a test.", "This is another test."]
        file_names = ["file1.txt", "file2.txt"]
        duplicate_pairs, duplicate_flags = detect_duplicates(file_texts, file_names)
        self.assertEqual(duplicate_pairs, [])
        self.assertEqual(duplicate_flags, {"file1.txt": False, "file2.txt": False})

if __name__ == '__main__':
    unittest.main()