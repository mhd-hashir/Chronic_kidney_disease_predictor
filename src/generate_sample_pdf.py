from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_sample_pdf(filename="sample_patient_report.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "City General Hospital")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, "123 Medical Drive, New York, NY")
    c.drawString(50, height - 85, "Patient Laboratory Report")
    
    c.line(50, height - 100, width - 50, height - 100)
    
    # Patient Info
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 130, "Patient Information:")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 150, "Name: John Doe")
    c.drawString(50, height - 170, "Age: 55")
    c.drawString(300, height - 170, "Gender: Male")
    
    # Vitals Section (Text format matches regex patterns in document_parser.py)
    y = height - 220
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Clinical Vitals & Blood Work:")
    
    y -= 30
    c.setFont("Helvetica", 12)
    
    vitals = [
        "Blood Pressure: 80",
        "Specific Gravity: 1.020",
        "Albumin: 1",
        "Sugar: 0",
        "Blood Glucose: 121",
        "Blood Urea: 36",
        "Serum Creatinine: 1.2",
        "Sodium: 138",
        "Potassium: 4.5",
        "Hemoglobin: 15.4",
        "Packed Cell Volume: 44",
        "White Blood Cell: 7800",
        "Red Blood Cell: 5.2",
    ]
    
    for vital in vitals:
        c.drawString(50, y, vital)
        y -= 20
        
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Diagnosis / History:")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Hypertension: Yes")
    c.drawString(50, y - 20, "Diabetes: Yes")
    c.drawString(50, y - 40, "Coronary Artery Disease: No")
    c.drawString(50, y - 60, "Anemia: No")
    
    c.save()
    print(f"PDF generated: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_sample_pdf()
