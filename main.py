import os
import json
import time
from typing import List, Dict, Any

# Import your existing heading extractor
from chkmain import HeadingOnlyPDFExtractor

# Import the new modules
from section_extractor import SectionExtractor
from relevance_scorer import RelevanceScorer
from section_ranker import SectionRanker
from output import OutputFormatter

class PersonaDrivenDocumentAnalyzer:
    def __init__(self):
        self.heading_extractor = HeadingOnlyPDFExtractor()
        self.section_extractor = SectionExtractor(self.heading_extractor)
        self.relevance_scorer = RelevanceScorer()
        self.section_ranker = SectionRanker()
        self.output_formatter = OutputFormatter()
    
    def load_input_config(self, input_dir: str) -> Dict[str, Any]:
        """Load input configuration from input.json file"""
        input_json_path = os.path.join(input_dir, "input.json")
        
        if not os.path.exists(input_json_path):
            raise FileNotFoundError(f"input.json not found in '{input_dir}' directory")
        
        try:
            with open(input_json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate required fields
            if 'documents' not in config:
                raise ValueError("input.json must contain 'documents' array")
            
            if not isinstance(config['documents'], list) or len(config['documents']) == 0:
                raise ValueError("'documents' must be a non-empty array")
            
            # Extract filenames from documents
            pdf_filenames = []
            for doc in config['documents']:
                if 'filename' not in doc:
                    raise ValueError("Each document must have a 'filename' field")
                pdf_filenames.append(doc['filename'])
            
            print(f"📋 Loaded input.json with {len(pdf_filenames)} documents:")
            for i, doc in enumerate(config['documents'], 1):
                filename = doc.get('filename', 'N/A')
                title = doc.get('title', 'No title')
                print(f"  {i}. {filename} - {title}")
            
            return {
                'config': config,
                'pdf_filenames': pdf_filenames,
                'persona': config.get('persona', {}).get('role', ''),
                'job_to_be_done': config.get('job_to_be_done', {}).get('task', '')
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input.json: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading input.json: {str(e)}")
    
    def process_documents(self, 
                         input_dir: str, 
                         persona: str = None, 
                         job_to_be_done: str = None,
                         output_path: str = None) -> Dict[str, Any]:
        """Main processing function for Round 1B"""
        
        start_time = time.time()
        
        # Validate inputs
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory '{input_dir}' not found")
        
        # Load configuration from input.json
        try:
            input_config = self.load_input_config(input_dir)
            pdf_filenames = input_config['pdf_filenames']
            
            # Use persona and job from input.json if not provided as parameters
            if not persona:
                persona = input_config['persona']
            if not job_to_be_done:
                job_to_be_done = input_config['job_to_be_done']
                
        except Exception as e:
            print(f"❌ Error loading input.json: {str(e)}")
            print("Falling back to scanning directory for PDF files...")
            
            # Fallback: scan directory for PDF files
            pdf_filenames = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
            if not pdf_filenames:
                raise FileNotFoundError(f"No PDF files found in '{input_dir}'")
        
        # Validate that specified PDFs exist in the directory
        missing_files = []
        existing_files = []
        
        for filename in pdf_filenames:
            pdf_path = os.path.join(input_dir, filename)
            if os.path.exists(pdf_path):
                existing_files.append(filename)
            else:
                missing_files.append(filename)
        
        if missing_files:
            print(f"⚠️  Warning: The following files from input.json were not found:")
            for missing_file in missing_files:
                print(f"   - {missing_file}")
        
        if not existing_files:
            raise FileNotFoundError(f"None of the specified PDF files were found in '{input_dir}'")
        
        pdf_files = existing_files
        
        print(f"📂 Processing {len(pdf_files)} PDF files: {pdf_files}")
        print(f"👤 Persona: {persona}")
        print(f"🎯 Job: {job_to_be_done}")
        print("-" * 60)
        
        # Step 1: Extract sections from all documents
        all_sections = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_dir, pdf_file)
            print(f"📄 Processing {pdf_file}...")
            
            try:
                sections = self.section_extractor.extract_sections_with_content(pdf_path)
                print(f"  ✅ Extracted {len(sections)} sections")
                all_sections.extend(sections)
            except Exception as e:
                print(f"  ❌ Error processing {pdf_file}: {str(e)}")
                continue
        
        if not all_sections:
            raise ValueError("No sections could be extracted from any document")
        
        print(f"\n📊 Total sections extracted: {len(all_sections)}")
        
        # Step 2: Score and rank sections by relevance
        print("🔍 Scoring sections for relevance...")
        ranked_sections = self.relevance_scorer.rank_sections(
            all_sections, persona, job_to_be_done
        )
        
        # Step 3: Apply additional filtering and ranking
        print("🎯 Applying advanced filtering...")
        filtered_sections = self.section_ranker.filter_and_deduplicate_sections(ranked_sections)
        balanced_sections = self.section_ranker.balance_section_distribution(filtered_sections)
        
        # Step 4: Create sub-section analysis
        print("📝 Creating sub-section analysis...")
        sub_section_analysis = self.section_ranker.create_sub_section_analysis(balanced_sections)
        
        # Step 5: Format output
        print("📋 Formatting output...")
        output_data = self.output_formatter.format_output(
            input_documents=pdf_files,
            persona=persona,
            job_to_be_done=job_to_be_done,
            extracted_sections=balanced_sections,
            sub_section_analysis=sub_section_analysis
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        output_data["processing_summary"]["processing_time_seconds"] = round(processing_time, 2)
        
        # Save output if path provided
        if output_path:
            self.output_formatter.save_output(output_data, output_path)
        
        # Print summary
        self.output_formatter.print_summary(output_data)
        
        print(f"\n⏱️  Processing completed in {processing_time:.2f} seconds")
        
        return output_data

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Create default config
        default_config = {
            "persona": "PhD Researcher in Computational Biology",
            "job_to_be_1done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
            "input_directory": "input",
            "output_directory": "output"
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"📝 Created default config file: {config_path}")
        print("Please update the config file with your specific persona and job requirements.")
        return default_config

def interactive_setup() -> Dict[str, str]:
    """Interactive setup for persona and job"""
    print("\n🚀 PERSONA-DRIVEN DOCUMENT ANALYZER")
    print("=" * 50)
    
    # Sample personas and jobs for reference
    sample_cases = {
        "1": {
            "persona": "PhD Researcher in Computational Biology",
            "job": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
        },
        "2": {
            "persona": "Investment Analyst",
            "job": "Analyze revenue trends, R&D investments, and market positioning strategies"
        },
        "3": {
            "persona": "Undergraduate Chemistry Student",
            "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
        }
    }
    
    print("\nSample test cases:")
    for key, case in sample_cases.items():
        print(f"{key}. {case['persona']}")
        print(f"   Job: {case['job']}")
    
    choice = input("\nUse sample case (1-3) or custom (c)? [1]: ").strip() or "1"
    
    if choice in sample_cases:
        selected = sample_cases[choice]
        return {
            "persona": selected["persona"],
            "job_to_be_done": selected["job"]
        }
    else:
        persona = input("Enter persona (role and expertise): ").strip()
        job = input("Enter job-to-be-done: ").strip()
        
        if not persona or not job:
            print("Using default values...")
            return sample_cases["1"]
        
        return {
            "persona": persona,
            "job_to_be_done": job
        }

def main():
    """Main function for Round 1B challenge"""
    
    # Setup directories
    input_dir = "input"
    output_dir = "output"
    
    # Create directories if they don't exist
    if not os.path.exists(input_dir):
        print(f"📁 Creating '{input_dir}' directory...")
        os.makedirs(input_dir)
        print(f"Please add your PDF files and input.json to the '{input_dir}' directory.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for input.json first
    input_json_path = os.path.join(input_dir, "input.json")
    if not os.path.exists(input_json_path):
        print(f"❌ input.json not found in '{input_dir}' directory.")
        print("Please create an input.json file with document specifications.")
        print("Example format:")
        print('''{
    "documents": [
        {
            "filename": "document1.pdf",
            "title": "Document Title"
        }
    ],
    "persona": {
        "role": "PhD Researcher in Computational Biology"
    },
    "job_to_be_done": {
        "task": "Prepare a comprehensive literature review"
    }
}''')
        return
    
    try:
        # Initialize analyzer
        analyzer = PersonaDrivenDocumentAnalyzer()
        
        # Load input configuration (persona and job will be read from input.json)
        print("📋 Reading configuration from input.json...")
        
        # Process documents - persona and job will be extracted from input.json
        output_path = os.path.join(output_dir, "challenge1b_output.json")
        result = analyzer.process_documents(
            input_dir=input_dir,
            output_path=output_path
        )
        
        print(f"\n🎉 Analysis complete! Check '{output_path}' for detailed results.")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()