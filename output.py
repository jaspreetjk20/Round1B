# output.py
import json
import time
from typing import Dict, Any, List

class OutputFormatter:
    def format_output(self, input_documents: List[str], persona: str, 
                     job_to_be_done: str, extracted_sections: List[Dict], 
                     sub_section_analysis: List[Dict]) -> Dict[str, Any]:
        return {
            "analysis_metadata": {
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "input_documents": input_documents,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "total_sections": len(extracted_sections)
            },
            "ranked_sections": extracted_sections,
            "sub_section_analysis": sub_section_analysis,
            "processing_summary": {
                "sections_analyzed": len(extracted_sections),
                "documents_processed": len(input_documents)
            }
        }
    
    def save_output(self, output_data: Dict[str, Any], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Output saved to {output_path}")
    
    def print_summary(self, output_data: Dict[str, Any]):
        metadata = output_data.get("analysis_metadata", {})
        sections = output_data.get("ranked_sections", [])
        
        print(f"\n📊 ANALYSIS SUMMARY")
        print(f"👤 Persona: {metadata.get('persona', 'N/A')}")
        print(f"📚 Documents: {len(metadata.get('input_documents', []))}")
        print(f"📄 Sections: {len(sections)}")
        
        if sections:
            print(f"\n🏆 Top 5 Most Relevant Sections:")
            for i, section in enumerate(sections[:5], 1):
                title = section.get('section_title', 'Untitled')[:50]
                score = section.get('relevance_score', 0)
                doc = section.get('document', 'Unknown')
                print(f"  {i}. {title}... (Score: {score:.3f}, Doc: {doc})")