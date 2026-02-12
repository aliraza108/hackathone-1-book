#!/usr/bin/env python3
"""
Script to process the textbook PDF and generate embeddings
"""
import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.app.services.pdf_processor import PDFProcessor
from backend.app.config import Config

async def main():
    # Initialize the PDF processor
    processor = PDFProcessor()
    
    # Path to the textbook PDF
    pdf_path = "content.pdf"  # The PDF in the root directory
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return
    
    print(f"Processing PDF: {pdf_path}")
    
    try:
        # Process the PDF and generate embeddings
        result = await processor.process_pdf(pdf_path)
        print(f"Successfully processed {result['chunks_processed']} content chunks")
        
        # Check the processing status
        status = await processor.get_processing_status()
        print(f"Processing status: {status}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())