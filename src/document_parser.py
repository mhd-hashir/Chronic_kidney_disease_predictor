import io
import re
from pypdf import PdfReader

class DocumentParser:
    def parse_pdf(self, file_bytes):
        """
        Extracts text from a PDF file stream and parses key medical vitals.
        Returns a dictionary of potential values.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return self._extract_from_text(text)
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return {}

    def _extract_from_text(self, text):
        """
        Uses RegEx to find key-value pairs in the text.
        Expected format in report: "Age: 45", "Hemoglobin: 12.5", etc.
        """
        data = {}
        text_lower = text.lower() # Normalize case for easier matching
        
        # Regex Patterns (Simplified for demo)
        patterns = {
            "age": r"age[:\s]+(\d+)",
            "bp": r"blood pressure[:\s]+(\d+)",
            "sg": r"specific gravity[:\s]+(1\.0\d+)",
            "al": r"albumin[:\s]+(\d)",
            "su": r"sugar[:\s]+(\d)",
            "hemo": r"hemoglobin[:\s]+(\d+\.?\d*)",
            "bgr": r"blood glucose[:\s]+(\d+)",
            "bu": r"blood urea[:\s]+(\d+)",
            "sc": r"serum creatinine[:\s]+(\d+\.?\d*)",
            "sod": r"sodium[:\s]+(\d+)",
            "pot": r"potassium[:\s]+(\d+\.?\d*)",
            "wc": r"white blood cell[:\s]+(\d+)",
            "rc": r"red blood cell[:\s]+(\d+\.?\d*)",
            # Boolean/Categorical logic is harder with regex, requires NLP or known keywords
            # For this demo, we'll try to match simple "Hypertension: Yes" style
            "htn": r"hypertension[:\s]+(yes|no)",
            "dm": r"diabetes[:\s]+(yes|no)",
            "ane": r"anemia[:\s]+(yes|no)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                val = match.group(1)
                # Convert to number if possible
                try:
                    if '.' in val:
                        data[key] = float(val)
                    elif val.isdigit():
                        data[key] = int(val)
                    else:
                        data[key] = val # Keep as string (yes/no)
                except:
                    data[key] = val
                    
        return data

# Test block
if __name__ == "__main__":
    pass
