from src.document_parser import DocumentParser
import os

def test_parser():
    file_path = "sample_patient_report.pdf"
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return

    print(f"Testing parser on {file_path}...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parser = DocumentParser()
    extracted_data = parser.parse_pdf(file_bytes)
    
    print("\nExtracted Data:")
    for key, value in extracted_data.items():
        print(f"{key}: {value}")

    # Validation
    expected_keys = ['age', 'bp', 'sg', 'hemo']
    missing = [k for k in expected_keys if k not in extracted_data]
    
    if not missing:
        print("\nSUCCESS: Critical vitals extracted.")
    else:
        print(f"\nFAILURE: Missing keys: {missing}")

if __name__ == "__main__":
    test_parser()
