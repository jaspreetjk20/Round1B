# Persona-Driven Document Analyzer

A sophisticated AI-powered document analysis system that extracts, analyzes, and ranks sections from PDF documents based on user personas and specific job requirements. This project implements intelligent content filtering and relevance scoring to deliver personalized document insights.

## 🚀 Quick Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jaspreetjk20/adobe-r1b-team-gud.git
   cd adobe-r1b-team-gud
   ```

2. **Create Virtual Environment & Install dependencies:**
   ```bash
    python -m venv venv
   pip install -r requirements.txt
    source venv/bin/activate
   ```

   Note: if you get an error saying "...cannot find module fronted...", then do execute the following : 

    ```bash
    pip uninstall pymupdf -y
    pip install pymupdf
    ```

3. **Prepare your input:**
   - Create an `input/` directory
   - Place your PDF files in the `input/` directory
   - Create an `input.json` file following the format below

4. **Run the analyzer:**
   ```bash
   python main.py
   ```

### Docker Deployment

1. **Build the Docker image:**
   ```bash
   docker build -t persona-doc-analyzer .
   ```

2. **Run the container:**
   ```bash
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none adober1b:latest
   ```

## 📁 Input Format

Create an `input.json` file in your `input/` directory:

```json
{
  "challenge_info": {
    "challenge_id": "your_analysis_id",
    "test_case_name": "your_case_name",
    "description": "Brief description"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document 1 Title"
    },
    {
      "filename": "document2.pdf", 
      "title": "Document 2 Title"
    }
  ],
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
  }
}
```

## 📖 Project Overview

The Persona-Driven Document Analyzer is an advanced AI system designed to revolutionize how users interact with large document collections. Built for Adobe's Round 1B challenge, this tool addresses the critical problem of information overload by providing intelligent, persona-specific document analysis.

### Core Functionality

The system operates through a sophisticated multi-stage pipeline that transforms raw PDF documents into highly relevant, ranked content sections. At its heart, the analyzer employs machine learning techniques combined with rule-based heuristics to understand both document structure and user intent.

**Document Processing Pipeline:**
The system begins with intelligent PDF parsing using PyMuPDF and pdfplumber libraries, extracting not just text but preserving document structure through heading detection. The `HeadingOnlyPDFExtractor` component uses advanced font analysis and layout detection to identify hierarchical document structures, enabling section-aware content extraction.

**Section Intelligence:**
The `SectionExtractor` component goes beyond simple text extraction, creating meaningful content segments that respect document boundaries and logical flow. Each section is enriched with metadata including confidence scores, word counts, and structural relationships.

**Relevance Scoring Engine:**
The heart of the system lies in its dual-approach relevance scoring. The `RelevanceScorer` component combines machine learning-based keyword expansion with domain-specific heuristics. Using TF-IDF vectorization and cosine similarity, the system identifies semantically related terms and concepts, while rule-based expansion ensures comprehensive coverage of domain-specific terminology.

**Intelligent Ranking:**
The `SectionRanker` applies sophisticated filtering algorithms that consider document domain detection, content quality assessment, and balanced representation across source documents. This ensures users receive diverse, high-quality content that addresses their specific needs without overwhelming them with redundant information.

### Key Features

- **Multi-Modal PDF Analysis**: Processes complex PDF layouts with tables, multi-column text, and varied formatting
- **Persona-Aware Content Filtering**: Adapts analysis based on user roles (researcher, analyst, student, etc.)
- **Job-Specific Relevance**: Tailors content ranking to specific tasks and objectives
- **Machine Learning Enhancement**: Employs lightweight ML models for keyword expansion and semantic similarity
- **Domain Intelligence**: Automatically detects document domains (academic, business, technical) for optimized processing
- **Quality Assurance**: Implements content deduplication, noise filtering, and informativeness scoring

### Technical Architecture

The system is built with modularity and scalability in mind, featuring clean separation of concerns across specialized components. The architecture supports both standalone execution and containerized deployment, making it suitable for various operational environments.

The codebase demonstrates modern Python practices with comprehensive type hints, robust error handling, and efficient processing algorithms. Memory usage is optimized through streaming PDF processing and intelligent content caching.

### Use Cases

This tool excels in scenarios requiring rapid comprehension of large document collections, including academic literature reviews, business intelligence gathering, legal document analysis, and educational content curation. The persona-driven approach ensures that a PhD researcher analyzing papers receives different insights than an undergraduate student studying the same materials.

## 📊 Output

The system generates a comprehensive JSON output containing:
- **Analysis Metadata**: Persona, job description, processing statistics
- **Ranked Sections**: Relevance-scored content sections with source attribution
- **Sub-section Analysis**: Detailed breakdown of most relevant content segments
- **Processing Summary**: Performance metrics and document coverage statistics

## 🧪 Testing

The project includes three test collections demonstrating different use cases:
- **Collection 1**: Travel planning documents for France
- **Collection 2**: Adobe Acrobat learning materials  
- **Collection 3**: Recipe and cooking guidance documents

Run tests using the provided test cases in the `test_cases/` directory.

## 🤝 Team GUD

This project was developed for Adobe India Hackathon's Round 1B challenge by Team GUD. 

- [Jaspreet Kaur](https://github.com/jaspreetjk20)
- [Anish Goenka](https://github.com/phANTom2303)
- [Anuranjan Singh Parihar](https://github.com/Anuranj-bot)