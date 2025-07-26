import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter
import math

# Import the ML keyword expander
try:
    from ml_keyword_expander import MLKeywordExpander
except ImportError:
    print("⚠️  ML keyword expander not available. Using rule-based approach only.")
    MLKeywordExpander = None

class RelevanceScorer:
    def __init__(self):
        self.persona_keywords = {}
        self.job_keywords = {}
        self.domain_specific_terms = {}
        
        # Initialize ML keyword expander if available
        self.ml_expander = None
        if MLKeywordExpander:
            try:
                self.ml_expander = MLKeywordExpander()
                print("✅ ML keyword expander initialized")
            except Exception as e:
                print(f"⚠️  ML expander initialization failed: {e}")
                self.ml_expander = None
        
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'mine', 'yours', 'ours', 'theirs', 'myself', 'yourself', 'himself',
            'herself', 'itself', 'ourselves', 'yourselves', 'themselves'
        }
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Return most common meaningful terms
        return [word for word, count in Counter(keywords).most_common(50)]
    
    def _fallback_keyword_expansion(self, persona_keywords: List[str], job_keywords: List[str]):
        """Fallback method for keyword expansion when ML is not available"""
        # Define domain-specific keyword mappings
        domain_mappings = {
            # Academic/Research terms
            'research': ['methodology', 'analysis', 'study', 'experiment', 'results', 'conclusion', 'literature', 'review', 'findings', 'data'],
            'student': ['concept', 'theory', 'principle', 'example', 'definition', 'explanation', 'summary', 'key', 'important', 'exam'],
            'phd': ['methodology', 'literature', 'review', 'analysis', 'benchmarks', 'performance', 'evaluation', 'datasets'],
            'undergraduate': ['basics', 'fundamentals', 'introduction', 'overview', 'concepts', 'examples', 'practice'],
            
            # Business terms
            'analyst': ['trends', 'analysis', 'financial', 'revenue', 'growth', 'market', 'performance', 'metrics', 'data'],
            'investment': ['revenue', 'profit', 'growth', 'market', 'financial', 'earnings', 'performance', 'strategy'],
            'business': ['strategy', 'market', 'revenue', 'growth', 'analysis', 'trends', 'performance', 'competitive'],
            
            # Technical terms
            'neural': ['network', 'model', 'training', 'performance', 'accuracy', 'dataset', 'algorithm', 'method'],
            'chemistry': ['reaction', 'mechanism', 'compound', 'synthesis', 'analysis', 'properties', 'structure'],
            'organic': ['synthesis', 'mechanism', 'reaction', 'compound', 'structure', 'properties', 'kinetics'],
            
            # Job-specific terms
            'literature': ['review', 'methodology', 'analysis', 'findings', 'results', 'study', 'research'],
            'exam': ['key', 'important', 'concept', 'theory', 'principle', 'definition', 'summary'],
            'financial': ['revenue', 'profit', 'earnings', 'growth', 'performance', 'analysis', 'trends'],
        }
        
        # Expand keywords based on domain mappings
        expanded_persona = set(persona_keywords)
        expanded_job = set(job_keywords)
        
        for keyword in persona_keywords + job_keywords:
            if keyword in domain_mappings:
                expanded_persona.update(domain_mappings[keyword][:5])  # Add top 5 related terms
                expanded_job.update(domain_mappings[keyword][:5])
        
        self.persona_keywords = list(expanded_persona)
        self.job_keywords = list(expanded_job)
        
        print(f"🔧 Rule-based expansion: {len(self.persona_keywords)} persona, {len(self.job_keywords)} job keywords")

    def analyze_persona_and_job(self, persona: str, job_to_be_done: str):
        """Analyze persona and job to extract relevant keywords and patterns"""
        
        # Extract keywords from persona and job
        persona_keywords = self.extract_keywords_from_text(persona)
        job_keywords = self.extract_keywords_from_text(job_to_be_done)
        
        # Use ML expansion if available
        if self.ml_expander:
            try:
                # Get ML-expanded keywords
                expanded_persona = self.ml_expander.expand_keywords(persona_keywords[:10], max_expansions=15)
                expanded_job = self.ml_expander.expand_keywords(job_keywords[:10], max_expansions=15)
                
                # Get domain-specific terms
                combined_text = persona + " " + job_to_be_done
                domain_terms = self.ml_expander.get_domain_specific_terms(combined_text)
                
                # Combine all keywords
                self.persona_keywords = list(set(persona_keywords + expanded_persona + domain_terms))
                self.job_keywords = list(set(job_keywords + expanded_job + domain_terms))
                
                print(f"🤖 ML-expanded keywords: {len(self.persona_keywords)} persona, {len(self.job_keywords)} job")
                
            except Exception as e:
                print(f"⚠️  ML expansion failed: {e}. Using rule-based approach.")
                self._fallback_keyword_expansion(persona_keywords, job_keywords)
        else:
            self._fallback_keyword_expansion(persona_keywords, job_keywords)
        
        # Store for debugging
        self.original_persona = persona
        self.original_job = job_to_be_done
    
    def calculate_keyword_similarity(self, text: str, keywords: List[str]) -> float:
        """Calculate TF-IDF-like similarity between text and keywords"""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        text_words = re.findall(r'\b\w+\b', text_lower)
        
        if not text_words:
            return 0.0
        
        # Count keyword occurrences
        keyword_matches = 0
        total_keyword_weight = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Count exact matches
            exact_matches = text_lower.count(keyword_lower)
            
            # Count partial matches (for compound terms)
            partial_matches = sum(1 for word in text_words if keyword_lower in word or word in keyword_lower)
            
            matches = exact_matches + (partial_matches * 0.5)  # Partial matches get half weight
            keyword_matches += matches
            total_keyword_weight += 1
        
        # Normalize by text length and keyword count
        similarity = (keyword_matches / len(text_words)) * (keyword_matches / total_keyword_weight)
        return min(similarity, 1.0)  # Cap at 1.0
    
    def calculate_section_importance(self, section: str, title: str) -> float:
        """Calculate importance based on section characteristics"""
        importance = 0.0
        
        # Length factor (moderate length is often more important)
        word_count = len(section.split())
        if 50 <= word_count <= 500:
            importance += 0.3
        elif 500 < word_count <= 1000:
            importance += 0.2
        elif word_count > 1000:
            importance += 0.1
        
        # Title importance indicators
        important_title_patterns = [
            r'\b(introduction|overview|summary|conclusion|abstract|methodology|results|analysis|discussion)\b',
            r'\b(key|important|main|primary|essential|fundamental|critical)\b',
            r'\b(chapter|section)\s+\d+',
            r'^\d+[\.\)]\s*\w+'
        ]
        
        title_lower = title.lower()
        for pattern in important_title_patterns:
            if re.search(pattern, title_lower):
                importance += 0.2
                break
        
        # Content quality indicators
        quality_indicators = [
            (r'\b(definition|define|concept|principle|theory)\b', 0.1),
            (r'\b(example|instance|case|illustration)\b', 0.1),
            (r'\b(result|finding|conclusion|outcome)\b', 0.15),
            (r'\b(method|approach|technique|procedure)\b', 0.1),
            (r'\b(analysis|evaluation|assessment|comparison)\b', 0.1),
            (r'\b(data|evidence|proof|statistics)\b', 0.1),
        ]
        
        section_lower = section.lower()
        for pattern, weight in quality_indicators:
            if re.search(pattern, section_lower):
                importance += weight
        
        return min(importance, 1.0)  # Cap at 1.0
    
    def score_section_relevance(self, section: Dict[str, Any]) -> float:
        """Score a section's relevance to the persona and job"""
        content = section.get('content', '')
        title = section.get('section_title', '')
        
        # Calculate different relevance components
        persona_relevance = self.calculate_keyword_similarity(content + ' ' + title, self.persona_keywords)
        job_relevance = self.calculate_keyword_similarity(content + ' ' + title, self.job_keywords)
        section_importance = self.calculate_section_importance(content, title)
        
        # Weight different factors
        persona_weight = 0.4
        job_weight = 0.4
        importance_weight = 0.2
        
        # Calculate final score
        final_score = (
            persona_relevance * persona_weight +
            job_relevance * job_weight +
            section_importance * importance_weight
        )
        
        # Boost score for certain section types based on job
        if any(keyword in self.original_job.lower() for keyword in ['literature', 'review', 'methodology']):
            if any(term in title.lower() for term in ['method', 'literature', 'review', 'approach']):
                final_score *= 1.2
        
        if any(keyword in self.original_job.lower() for keyword in ['exam', 'study', 'concept']):
            if any(term in title.lower() for term in ['concept', 'theory', 'principle', 'key']):
                final_score *= 1.2
        
        if any(keyword in self.original_job.lower() for keyword in ['financial', 'revenue', 'analysis']):
            if any(term in title.lower() for term in ['financial', 'revenue', 'analysis', 'performance']):
                final_score *= 1.2
        
        return min(final_score, 1.0)  # Cap at 1.0
    
    def rank_sections(self, sections: List[Dict[str, Any]], persona: str, job_to_be_done: str) -> List[Dict[str, Any]]:
        """Rank sections by relevance to persona and job"""
        
        # Analyze persona and job
        self.analyze_persona_and_job(persona, job_to_be_done)
        
        # Score each section
        scored_sections = []
        for section in sections:
            relevance_score = self.score_section_relevance(section)
            section_with_score = section.copy()
            section_with_score['relevance_score'] = relevance_score
            scored_sections.append(section_with_score)
        
        # Sort by relevance score (descending)
        ranked_sections = sorted(scored_sections, key=lambda x: x['relevance_score'], reverse=True)
        
        # Add importance rank
        for i, section in enumerate(ranked_sections):
            section['importance_rank'] = i + 1
        
        return ranked_sections