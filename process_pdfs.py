import os
import json
import tabula
import fitz  # PyMuPDF
from PIL import Image # Pillow for image processing
import io # Import io for image saving
import pandas as pd # Import pandas for tabula-py DataFrames
from pdfminer.high_level import extract_text as pdfminer_extract_text # Import pdfminer for PDF to TXT

# --- Configuration ---
# Directory where your original PDF files are located (and where .txt files are)
# IMPORTANT: Ensure this path is correct for YOUR system
PDF_INPUT_DIR = r"C:\Users\drris\Downloads\SLapp\pdfs"
# Directory where processed JSON data and extracted images will be saved
OUTPUT_DIR = r"C:\Users\drris\Downloads\SLapp\pdfs\processed_data"
# Directory for extracted images (will be created inside OUTPUT_DIR)
IMAGES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "extracted_images")

# --- Ensure output directories exist ---
print(f"Ensuring output directory exists: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Ensuring images output directory exists: {IMAGES_OUTPUT_DIR}")
os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)
print("Directories checked/created successfully.")

# --- Helper Functions ---

def extract_tables_from_pdf(pdf_file_path):
    """
    Extracts tables from a PDF using tabula-py and returns them as a list of dictionaries.
    Each dictionary represents a table, with keys as column headers.
    """
    tables_data = []
    if not os.path.exists(pdf_file_path):
        print(f"  - WARNING: Original PDF not found for table extraction at '{pdf_file_path}'. Skipping table extraction.")
        return tables_data
    try:
        # Pass the JVM option to suppress warnings
        java_opts = ["--enable-native-access=ALL-UNNAMED"]
        dfs = tabula.read_pdf(pdf_file_path, pages='all', multiple_tables=True, lattice=True, stream=False, encoding='utf-8', java_options=java_opts)
        for i, df in enumerate(dfs):
            tables_data.append({
                "table_id": f"table_{i+1}",
                "data": df.where(pd.notnull(df), None).to_dict(orient='records') # Replace NaN with None for JSON
            })
        print(f"  - Extracted {len(tables_data)} tables from {os.path.basename(pdf_file_path)}")
    except Exception as e:
        print(f"  - ERROR extracting tables from {os.path.basename(pdf_file_path)}: {e}")
        print("    (This often indicates a missing/incorrect Java installation or a malformed PDF. Ensure Java is in PATH.)")
    return tables_data

def extract_images_from_pdf(pdf_file_path, output_dir, doc_id):
    """
    Extracts images from a PDF using PyMuPDF and saves them as PNG files.
    Returns a list of paths to the saved images.
    """
    extracted_image_paths = []
    if not os.path.exists(pdf_file_path):
        print(f"  - WARNING: Original PDF not found for image extraction at '{pdf_file_path}'. Skipping image extraction.")
        return extracted_image_paths
    try:
        doc = fitz.open(pdf_file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"{doc_id}_page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)

                try:
                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(image_path)
                    extracted_image_paths.append(image_path)
                except Exception as img_e:
                    print(f"    - Could not save image {image_filename}: {img_e}")
        doc.close()
        print(f"  - Extracted {len(extracted_image_paths)} images from {os.path.basename(pdf_file_path)}")
    except Exception as e:
        print(f"  - ERROR extracting images from {os.path.basename(pdf_file_path)}: {e}")
    return extracted_image_paths


def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Breaks down text into smaller chunks with optional overlap.
    """
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start += (chunk_size - overlap)
    return chunks

# --- Main Processing Logic ---

all_processed_documents_metadata = []

# Define your document metadata with file paths
# IMPORTANT: These file_path entries must match the actual .txt files in C:\Users\drris\Downloads\SLapp\pdfs
# And original_pdf_path must match your actual .pdf files in C:\Users\drris\Downloads\SLapp\pdfs
# Ensure the .pdf files exist for table/image extraction!
documents_to_process = [
    {"id": "doc_1", "title": "1. tc19-16 - Energy Supply Device ARC Recommendation Report.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\1. tc19-16 - Energy Supply Device ARC Recommendation Report.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\1. tc19-16 - Energy Supply Device ARC Recommendation Report.pdf", "key_topics": ["Hydrogen Fuel Cell", "Safety", "Temperature Range"], "num_requirements": 2, "num_discrepancies": 1},
    {"id": "doc_2", "title": "2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.pdf", "key_topics": ["PEM Fuel Cell", "FMEA", "Leakage"], "num_requirements": 3, "num_discrepancies": 1},
    {"id": "doc_3", "title": "tc17-23 - Flammability of Materials in a Low-Concentration Hydrogen Environment.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc17-23 - Flammability of Materials in a Low-Concentration Hydrogen Environment.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc17-23 - Flammability of Materials in a Low-Concentration Hydrogen Environment.pdf", "key_topics": ["Flammability", "Materials", "Hydrogen Environment"], "num_requirements": 1, "num_discrepancies": 0},
    {"id": "doc_4", "title": "tc21-3 - Fuel Tank Flammability Assessment Method User’s Manual - Updated for Version 11.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc21-3 - Fuel Tank Flammability Assessment Method User's Manual - Updated for Version 11.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc21-3 - Fuel Tank Flammability Assessment Method User’s Manual - Updated for Version 11.pdf", "key_topics": ["Fuel Tank", "Flammability", "Assessment Method"], "num_requirements": 1, "num_discrepancies": 0},
    {"id": "doc_5", "title": "tc20-9 - Aircraft Fuel Cell and Safety Management System.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc20-9 - Aircraft Fuel Cell and Safety Management System.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc20-9 - Aircraft Fuel Cell and Safety Management System.pdf", "key_topics": ["Fuel Cell Safety", "Safety Management System", "Aircraft Systems"], "num_requirements": 2, "num_discrepancies": 0},
    {"id": "doc_6", "title": "tc19-55 - Aircraft Fuel Cell System.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-55 - Aircraft Fuel Cell System.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-55 - Aircraft Fuel Cell System.pdf", "key_topics": ["Aircraft Fuel Cell", "System Design"], "num_requirements": 1, "num_discrepancies": 0},
    {"id": "doc_7", "title": "tc19-17 - Evaluation for a Lightweight Fuel Cell Containment System for Aircraft Safety.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-17 - Evaluation for a Lightweight Fuel Cell Containment System for Aircraft Safety.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-17 - Evaluation for a Lightweight Fuel Cell Containment System for Aircraft Safety.pdf", "key_topics": ["Fuel Cell Containment", "Lightweight Systems", "Aircraft Safety"], "num_requirements": 1, "num_discrepancies": 0},
    {"id": "doc_8", "title": "tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.pdf", "key_topics": ["PEM Fuel Cell", "Abusive Testing", "Safety"], "num_requirements": 1, "num_discrepancies": 0},
    {"id": "doc_9", "title": "tc21-30 - Study of Unitized Regenerative Fuel Cell Systems for Aircraft Applications.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc21-30 - Study of Unitized Regenerative Fuel Cell Systems for Aircraft Applications.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc21-30 - Study of Unitized Regenerative Fuel Cell Systems for Aircraft Applications.pdf", "key_topics": ["Regenerative Fuel Cell", "Aircraft Applications"], "num_requirements": 1, "num_discrepancies": 0}#,
    # {"id": "doc_10", "title": "tc16-24.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc16-24.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc16-24.pdf", "key_topics": ["General Testing", "Safety"], "num_requirements": 1, "num_discrepancies": 0},
    # {"id": "doc_11", "title": "tc19-16.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-16.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc19-16.pdf", "key_topics": ["General Aircraft", "Maintenance"], "num_requirements": 0, "num_discrepancies": 0},
    # {"id": "doc_12", "title": "tc18-49.pdf", "file_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc18-49.txt", "original_pdf_path": r"C:\Users\drris\Downloads\SLapp\pdfs\tc18-49.pdf", "key_topics": ["General Fuel Cell", "Performance"], "num_requirements": 0, "num_discrepancies": 0}
]

# --- Main Processing Loop ---
for doc_info in documents_to_process:
    doc_id = doc_info["id"]
    doc_title = doc_info["title"]
    doc_txt_file_path = doc_info["file_path"]
    doc_pdf_file_path = doc_info["original_pdf_path"] # Use original PDF for tables/images

    print(f"\n--- Processing document: {doc_title} ---")

    # 1. Convert PDF to TXT using pdfminer.six and then read the content
    text_content = None
    if os.path.exists(doc_pdf_file_path):
        try:
            # Extract text directly from PDF
            text_content = pdfminer_extract_text(doc_pdf_file_path)
            # Save the extracted text to the .txt file path
            with open(doc_txt_file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"  - Successfully extracted text from PDF and saved to '{os.path.basename(doc_txt_file_path)}'")
        except Exception as e:
            print(f"  - ERROR extracting text from PDF '{os.path.basename(doc_pdf_file_path)}' using pdfminer.six: {e}. This document's text content will be empty.")
            text_content = "" # Set to empty string if extraction fails
    else:
        print(f"  - ERROR: Original PDF not found at '{doc_pdf_file_path}'. Cannot extract text. This document's text content will be empty.")
        text_content = ""


    if not text_content: # Check if text_content is empty after extraction attempt
        print(f"Skipping further processing (tables, images, chunks) for {doc_title} due to empty/unreadable text content.")
        # We still append metadata even if text is empty, so it shows up in the dashboard
        all_processed_documents_metadata.append({
            "id": doc_id,
            "title": doc_title,
            "original_txt_file_path": doc_txt_file_path,
            "original_pdf_file_path": doc_pdf_file_path,
            "full_text_content": "", # Explicitly empty
            "chunks": [],
            "extracted_tables": [],
            "extracted_image_paths": [],
            "key_topics": doc_info["key_topics"],
            "num_requirements": doc_info["num_requirements"],
            "num_discrepancies": doc_info["num_discrepancies"]
        })
        continue # Skip to next document


    # 2. Extract Tables (from original .pdf file)
    extracted_tables = extract_tables_from_pdf(doc_pdf_file_path)

    # 3. Extract Images (from original .pdf file)
    extracted_image_paths = extract_images_from_pdf(doc_pdf_file_path, IMAGES_OUTPUT_DIR, doc_id)

    # 4. Chunk Text
    chunks = chunk_text(text_content, chunk_size=500, overlap=50)
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        processed_chunks.append({
            "chunk_id": f"{doc_id}_chunk_{i+1}",
            "text": chunk,
            "length": len(chunk),
            "start_char": text_content.find(chunk)
        })

    # Add all extracted data to metadata
    all_processed_documents_metadata.append({
        "id": doc_id,
        "title": doc_title,
        "original_txt_file_path": doc_txt_file_path, # Path to the .txt version
        "original_pdf_file_path": doc_pdf_file_path, # Path to the original .pdf
        "full_text_content": text_content,
        "chunks": processed_chunks,
        "extracted_tables": extracted_tables,
        "extracted_image_paths": extracted_image_paths, # Store paths to saved images
        "key_topics": doc_info["key_topics"],
        "num_requirements": doc_info["num_requirements"],
        "num_discrepancies": doc_info["num_discrepancies"]
    })

# --- Save Processed Data to JSON Files ---
# These will be loaded by your app.py

# Save the comprehensive document metadata
print(f"\nSaving processed document metadata to {os.path.join(OUTPUT_DIR, 'processed_documents_metadata.json')}")
with open(os.path.join(OUTPUT_DIR, "processed_documents_metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(all_processed_documents_metadata, f, indent=4)
print("Saved processed_documents_metadata.json successfully.")

# Save the other simulated data (entities, checklist, discrepancies, QA)
# These are still hardcoded in this script for now, as their generation
# would involve actual AI models processing the extracted text/tables/images.
# In a real system, these would be generated dynamically.

# Placeholder for dynamically extracted entities (currently empty from this script)
dummy_extracted_entities = {
    "doc_1": [{"text": "hydrogen fuel cells", "type": "AIRCRAFT_SYSTEM"}, {"text": "-40C", "type": "TEMPERATURE"}],
    "doc_2": [{"text": "PEM fuel cell systems", "type": "AIRCRAFT_SYSTEM"}, {"text": "80C", "type": "TEMPERATURE"}],
    "doc_3": [{"text": "ASTM E1354", "type": "CERTIFICATION_STANDARD"}],
    "doc_4": [{"text": "fuel tank flammability", "type": "SAFETY_ASPECT"}],
    "doc_5": [{"text": "safety management system", "type": "SYSTEM"}],
    "doc_6": [{"text": "aircraft fuel cell system", "type": "AIRCRAFT_SYSTEM"}],
    "doc_7": [{"text": "lightweight fuel cell containment", "type": "COMPONENT"}],
    "doc_8": [{"text": "abusive testing", "type": "TEST_PROCEDURE"}],
    "doc_9": [{"text": "tc16-24", "type": "DOCUMENT_ID"}],
    "doc_10": [{"text": "regenerative fuel cell", "type": "FUEL_CELL_TYPE"}],
    "doc_11": [{"text": "general aircraft", "type": "AIRCRAFT_TYPE"}],
    "doc_12": [{"text": "general fuel cell", "type": "FUEL_CELL_TYPE"}]
}
print(f"Saving extracted entities (simulated) to {os.path.join(OUTPUT_DIR, 'extracted_entities.json')}")
with open(os.path.join(OUTPUT_DIR, "extracted_entities.json"), 'w', encoding='utf-8') as f:
    json.dump(dummy_extracted_entities, f, indent=4)
print("Saved extracted_entities.json successfully.")

# Hardcoded fake checklist data (replace with AI-generated in real app)
print(f"Saving checklist items (simulated) to {os.path.join(OUTPUT_DIR, 'checklist_items.json')}")
fake_checklist_data = [
    {"id": "req_001", "description": "Fuel cells must operate safely between -40°C and +85°C.", "related_docs": ["doc_1"], "relevant_section": "Doc 1, Sec 3.1.2", "related_entities": ["Hydrogen Fuel Cell", "Temperature Range"], "status": "Pending", "notes": "", "dependencies": [], "associated_discrepancy": True},
    {"id": "req_002", "description": "Pressure relief valve per FAR Part 25.903(b).", "related_docs": ["doc_1", "doc_2"], "relevant_section": "Doc 1, Sec 3.1.2; Doc 2, Para 2", "related_entities": ["Pressure Relief Valve", "FAR Part 25.903(b)"], "status": "In Progress", "notes": "Eng review needed.", "dependencies": ["req_001"], "associated_discrepancy": True},
    {"id": "req_003", "description": "PEM fuel cell max temp 80°C.", "related_docs": ["doc_2"], "relevant_section": "Doc 2, Para 1", "related_entities": ["PEM Fuel Cell System", "Temperature Limit"], "status": "Pending", "notes": "", "dependencies": [], "associated_discrepancy": False},
    {"id": "req_004", "description": "Materials must meet fire resistance (ASTM E1354).", "related_docs": ["doc_3"], "relevant_section": "Doc 3, Para 1", "related_entities": ["Materials", "ASTM E1354"], "status": "Completed", "notes": "Report filed.", "dependencies": [], "associated_discrepancy": False},
    {"id": "req_005", "description": "Fuel tank flammability assessment must be performed.", "related_docs": ["doc_4"], "relevant_section": "Doc 4, Sec 2.1", "related_entities": ["Fuel Tank", "Flammability"], "status": "Pending", "notes": "", "dependencies": [], "associated_discrepancy": False},
    {"id": "req_006", "description": "Safety Management System must integrate fuel cell operations.", "related_docs": ["doc_5"], "relevant_section": "Doc 5, Ch 3", "related_entities": ["Safety Management System", "Fuel Cell Safety"], "status": "In Progress", "notes": "SOPs being drafted.", "dependencies": [], "associated_discrepancy": False},
    {"id": "req_007", "description": "Lightweight containment system evaluation is necessary for aircraft safety.", "related_docs": ["doc_7"], "relevant_section": "Doc 7, Sec 1", "related_entities": ["Lightweight Systems", "Aircraft Safety"], "status": "Pending", "notes": "", "dependencies": [], "associated_discrepancy": False}
]
with open(os.path.join(OUTPUT_DIR, "checklist_items.json"), 'w', encoding='utf-8') as f:
    json.dump(fake_checklist_data, f, indent=4)
print("Saved checklist_items.json successfully.")

# Hardcoded fake discrepancy data (replace with AI-generated in real app)
print(f"Saving discrepancies (simulated) to {os.path.join(OUTPUT_DIR, 'discrepancies.json')}")
fake_discrepancy_data = [
    {"id": "disc_001", "conflicting_requirements": ["req_001", "req_003"], "nature_of_conflict": "Temperature operating range discrepancy: Doc 1 allows up to +85°C, but Doc 2 states maximum 80°C for PEM systems. Need clarity on specific PEM fuel cell limits.", "involved_docs": ["doc_1", "doc_2"], "proposed_resolution_notes": "Proposed to follow 80C limit for all PEM systems to be conservative.", "clarity_request_status": "Drafted"},
    {"id": "disc_002", "conflicting_requirements": ["req_002"], "nature_of_conflict": "Pressure relief valve activation pressure: Doc 2 specifies 550 PSI, but Doc 1 (and FAR Part 25.903b) is general. Need specific activation pressure for this valve based on system design.", "involved_docs": ["doc_1", "doc_2"], "proposed_resolution_notes": "Awaiting design specifications before finalizing pressure setting.", "clarity_request_status": "Not Sent"}
]
with open(os.path.join(OUTPUT_DIR, "discrepancies.json"), 'w', encoding='utf-8') as f:
    json.dump(fake_discrepancy_data, f, indent=4)
print("Saved discrepancies.json successfully.")

# Hardcoded fake QA responses (replace with AI-generated in real app)
print(f"Saving QA responses (simulated) to {os.path.join(OUTPUT_DIR, 'qa_responses.json')}")
fake_qa_responses = [
    {"question": "what are the flammability requirements?", "answer": "Materials in hydrogen environments must meet fire resistance standards like ASTM E1354. Hydrogen concentrations exceeding 4% are considered hazardous for ignition. All wiring insulation also requires special attention.", "sources": ["doc_3"]},
    {"question": "what is the required temperature for fuel cells?", "answer": "Documents indicate a general operating range of -40°C to +85°C. For PEM fuel cell systems, the maximum operating temperature should not exceed 80°C.", "sources": ["doc_1", "doc_2"]},
    {"question": "tell me about fuel tank flammability.", "answer": "The fuel tank flammability assessment method is detailed in tc21-3 (Doc 4). It outlines procedures to evaluate the fire safety aspects of fuel tanks.", "sources": ["doc_4"]},
    {"question": "what is abusive testing?", "answer": "Abusive testing for Proton Exchange Membrane (PEM) Hydrogen Fuel Cells is covered in document tc16-24 (Doc 8). This testing evaluates their behavior under extreme conditions.", "sources": ["doc_8"]}
]
with open(os.path.join(OUTPUT_DIR, "qa_responses.json"), 'w', encoding='utf-8') as f:
    json.dump(fake_qa_responses, f, indent=4)
print("Saved qa_responses.json successfully.")

print("\nProcessing complete. Now, you can run app.py to see the dashboard with richer data.")