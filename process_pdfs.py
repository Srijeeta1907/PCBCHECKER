import pdfplumber
import pandas as pd
import os
import glob

# Find the PDF
pdf_files = glob.glob("data/raw_pdfs/*.pdf")

if not pdf_files:
    print("❌ Error: No PDF files found in data/raw_pdfs/!")
    exit()

pdf_path = pdf_files[0]
output_path = "data/processed_data/smd_database.csv"

def extract_smd_data():
    all_rows = []
    
    print(f"Reading {pdf_path}... this might take a minute.")
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # extract_table returns a list of lists (rows and columns)
            table = page.extract_table()
            
            if table:
                for row in table:
                    # Clean up the row: replace 'None' with empty strings and remove newlines
                    clean_row = [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
                    all_rows.append(clean_row)
            
            # Print progress every 10 pages so you know it's not frozen
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} pages...")
                
    if all_rows:
        # Convert the raw list of lists directly into a DataFrame
        final_df = pd.DataFrame(all_rows)
        
        # Drop completely empty rows
        final_df.dropna(how='all', inplace=True)
        
        # Ensure the directory exists
        os.makedirs("data/processed_data", exist_ok=True)
        
        # Save without headers to avoid the Pandas crash
        final_df.to_csv(output_path, index=False, header=False)
        
        print(f"✅ Success! Extracted {len(final_df)} rows.")
        print(f"💾 Data saved to {output_path}")
    else:
        print("❌ Could not find any tables in the PDF.")

if __name__ == "__main__":
    extract_smd_data()