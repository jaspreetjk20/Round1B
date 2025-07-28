import fitz
import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class SectionExtractor:
    def __init__(self, heading_extractor):
        self.heading_extractor = heading_extractor
        
    def extract_sections_with_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract sections with their full content"""
        # Get headings from your existing extractor
        title, headings = self.heading_extractor.process(pdf_path)
        
        if not headings:
            return self._extract_full_document_as_section(pdf_path)
        
        # Now extract content for each section
        doc = fitz.open(pdf_path)
        sections = []
        
        # Sort headings by page and position
        sorted_headings = sorted(headings, key=lambda x: (x['page'], x.get('y_position', 0)))
        
        for i, heading in enumerate(sorted_headings):
            section_content = self._extract_section_content(
                doc, heading, 
                sorted_headings[i+1] if i+1 < len(sorted_headings) else None
            )
            
            if section_content:
                sections.append({
                    'document': pdf_path.split('/')[-1].replace('.pdf', ''),
                    'section_title': heading['text'],
                    'page_number': heading['page'],
                    'content': section_content,
                    'heading_level': heading['level'],
                    'confidence': heading.get('confidence', 0),
                    'word_count': len(section_content.split()),
                    'start_page': heading['page']
                })
        
        doc.close()
        
        # If no sections found, extract document as single section
        if not sections:
            return self._extract_full_document_as_section(pdf_path)
            
        return sections
    
    def _extract_section_content(self, doc, current_heading: Dict, next_heading: Dict = None) -> str:
        """Extract content between current heading and next heading"""
        content_parts = []
        start_page = current_heading['page'] - 1  # PyMuPDF uses 0-based indexing
        
        if next_heading:
            end_page = next_heading['page'] - 1
        else:
            end_page = len(doc) - 1
        
        # Extract text from start_page to end_page
        for page_num in range(start_page, min(end_page + 1, len(doc))):
            page = doc[page_num]
            page_text = page.get_text()
            
            if page_num == start_page:
                # Find heading position and extract content after it
                heading_text = current_heading['text']
                heading_pos = page_text.find(heading_text)
                if heading_pos != -1:
                    page_text = page_text[heading_pos + len(heading_text):]
            
            if page_num == end_page and next_heading and next_heading['page'] == end_page + 1:
                # Stop before next heading if on same page
                next_heading_text = next_heading['text']
                next_heading_pos = page_text.find(next_heading_text)
                if next_heading_pos != -1:
                    page_text = page_text[:next_heading_pos]
            
            content_parts.append(page_text)
        
        # Clean and join content
        full_content = ' '.join(content_parts)
        return self._clean_extracted_text(full_content)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Clean up extra spaces and newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text
    
    def _extract_full_document_as_section(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Fallback: extract entire document as one section"""
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            full_text += page.get_text() + "\n"
        
        doc.close()
        
        return [{
            'document': pdf_path.split('/')[-1].replace('.pdf', ''),
            'section_title': 'Full Document',
            'page_number': 1,
            'content': self._clean_extracted_text(full_text),
            'heading_level': 'H1',
            'confidence': 1.0,
            'word_count': len(full_text.split()),
            'start_page': 1
        }]