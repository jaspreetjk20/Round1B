import re
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class SectionRanker:
    def __init__(self):
        self.refinement_rules = {
            'academic': {
                'key_phrases': ['methodology', 'analysis', 'results', 'findings', 'literature review'],
                'remove_patterns': [r'references?', r'bibliography', r'appendix'],
                'boost_patterns': [r'introduction', r'methodology', r'results', r'conclusion']
            },
            'business': {
                'key_phrases': ['revenue', 'financial', 'market', 'growth', 'performance'],
                'remove_patterns': [r'disclaimer', r'legal notice', r'copyright'],
                'boost_patterns': [r'executive summary', r'financial highlights', r'revenue']
            },
            'educational': {
                'key_phrases': ['concept', 'theory', 'principle', 'example', 'definition'],
                'remove_patterns': [r'exercise', r'homework', r'problem set'],
                'boost_patterns': [r'key concepts?', r'important', r'fundamental']
            }
        }
    
    def detect_document_domain(self, sections: List[Dict[str, Any]]) -> str:
        """Detect the primary domain of the document collection"""
        all_text = ' '.join([s.get('content', '') + ' ' + s.get('section_title', '') for s in sections])
        all_text = all_text.lower()
        
        domain_indicators = {
            'academic': ['research', 'study', 'analysis', 'methodology', 'literature', 'findings', 'experiment'],
            'business': ['revenue', 'financial', 'market', 'company', 'business', 'profit', 'investment'],
            'educational': ['chapter', 'concept', 'theory', 'principle', 'definition', 'example', 'student']
        }
        
        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_text)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
    
    def refine_section_text(self, section: Dict[str, Any], domain: str) -> str:
        """Refine section text based on domain and relevance"""
        content = section.get('content', '')
        title = section.get('section_title', '')
        
        if not content:
            return content
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        refined_sentences = []
        
        # Get domain-specific rules
        rules = self.refinement_rules.get(domain, self.refinement_rules['academic'])
        key_phrases = rules['key_phrases']
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Skip sentences that are likely not relevant
            if self._should_skip_sentence(sentence, rules['remove_patterns']):
                continue
            
            # Keep sentences with key phrases
            sentence_lower = sentence.lower()
            has_key_phrase = any(phrase in sentence_lower for phrase in key_phrases)
            
            # Keep sentences that seem informative
            is_informative = self._is_informative_sentence(sentence)
            
            if has_key_phrase or is_informative:
                refined_sentences.append(sentence)
        
        # If we filtered too much, keep original content
        refined_text = '. '.join(refined_sentences)
        if len(refined_text) < len(content) * 0.3:  # If we removed more than 70%
            return content[:1000] + '...' if len(content) > 1000 else content
        
        return refined_text
    
    def _should_skip_sentence(self, sentence: str, remove_patterns: List[str]) -> bool:
        """Check if sentence should be skipped"""
        sentence_lower = sentence.lower()
        
        # Skip if matches remove patterns
        for pattern in remove_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        # Skip if it's mostly numbers or symbols
        if len(re.sub(r'[^\w\s]', '', sentence)) < len(sentence) * 0.5:
            return True
        
        # Skip if it's a page reference or citation
        if re.match(r'^\s*(page|p\.)\s*\d+', sentence_lower):
            return True
        
        # Skip if it's mostly uppercase (likely headers/footers)
        if sentence.isupper() and len(sentence) > 10:
            return True
        
        return False
    
    def _is_informative_sentence(self, sentence: str) -> bool:
        """Check if sentence is likely to be informative"""
        sentence_lower = sentence.lower()
        
        # Contains informative verbs
        informative_verbs = ['shows', 'demonstrates', 'indicates', 'suggests', 'reveals', 'explains', 'describes', 'analyzes', 'examines']
        has_informative_verb = any(verb in sentence_lower for verb in informative_verbs)
        
        # Contains quantitative information
        has_numbers = bool(re.search(r'\d+', sentence))
        
        # Has reasonable length
        word_count = len(sentence.split())
        reasonable_length = 10 <= word_count <= 50
        
        # Contains subject matter keywords
        subject_keywords = ['method', 'result', 'finding', 'analysis', 'data', 'study', 'research', 'concept', 'theory']
        has_subject_keywords = any(keyword in sentence_lower for keyword in subject_keywords)
        
        # Score based on multiple factors
        score = 0
        if has_informative_verb: score += 2
        if has_numbers: score += 1
        if reasonable_length: score += 1
        if has_subject_keywords: score += 2
        
        return score >= 3
    
    def create_sub_section_analysis(self, ranked_sections: List[Dict[str, Any]], max_sections: int = 10) -> List[Dict[str, Any]]:
        """Create sub-section analysis with refined text"""
        
        # Detect domain for better refinement
        domain = self.detect_document_domain(ranked_sections)
        
        # Take top sections
        top_sections = ranked_sections[:max_sections]
        
        sub_section_analysis = []
        for section in top_sections:
            refined_text = self.refine_section_text(section, domain)
            
            # Create sub-section entry
            sub_section = {
                'document': section['document'],
                'section_title': section['section_title'],
                'refined_text': refined_text,
                'page_number': section['page_number'],
                'importance_rank': section['importance_rank'],
                'relevance_score': section.get('relevance_score', 0),
                'original_length': len(section.get('content', '')),
                'refined_length': len(refined_text),
                'domain': domain
            }
            
            sub_section_analysis.append(sub_section)
        
        return sub_section_analysis
    
    def filter_and_deduplicate_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar sections"""
        if not sections:
            return sections
        
        filtered_sections = []
        seen_titles = set()
        seen_content_hashes = set()
        
        for section in sections:
            title = section.get('section_title', '').lower().strip()
            content = section.get('content', '')
            
            # Skip if title is too similar to existing
            if any(self._titles_too_similar(title, seen_title) for seen_title in seen_titles):
                continue
            
            # Create simple content hash for duplicate detection
            content_words = re.findall(r'\w+', content.lower())
            if len(content_words) > 10:
                content_hash = hash(' '.join(sorted(content_words[:20])))  # Hash of first 20 unique words
                if content_hash in seen_content_hashes:
                    continue
                seen_content_hashes.add(content_hash)
            
            seen_titles.add(title)
            filtered_sections.append(section)
        
        return filtered_sections
    
    def _titles_too_similar(self, title1: str, title2: str) -> bool:
        """Check if two titles are too similar"""
        if not title1 or not title2:
            return False
        
        # Exact match
        if title1 == title2:
            return True
        
        # One is substring of another
        if title1 in title2 or title2 in title1:
            return True
        
        # High word overlap
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total_unique = len(words1.union(words2))
            similarity = overlap / total_unique
            
            if similarity > 0.7:  # 70% word overlap
                return True
        
        return False
    
    def balance_section_distribution(self, sections: List[Dict[str, Any]], max_per_document: int = 5) -> List[Dict[str, Any]]:
        """Ensure balanced representation across documents"""
        if not sections:
            return sections
        
        # Group by document
        doc_sections = defaultdict(list)
        for section in sections:
            doc_name = section.get('document', 'unknown')
            doc_sections[doc_name].append(section)
        
        # Take top sections from each document
        balanced_sections = []
        for doc_name, doc_section_list in doc_sections.items():
            # Sort by relevance score within document
            sorted_doc_sections = sorted(doc_section_list, key=lambda x: x.get('relevance_score', 0), reverse=True)
            balanced_sections.extend(sorted_doc_sections[:max_per_document])
        
        # Sort final list by relevance score and reassign ranks
        final_sections = sorted(balanced_sections, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Reassign importance ranks
        for i, section in enumerate(final_sections):
            section['importance_rank'] = i + 1
        
        return final_sections