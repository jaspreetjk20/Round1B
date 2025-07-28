import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import pickle
import os
from typing import List, Dict, Any

class MLKeywordExpander:
    """Lightweight ML-based keyword expander using TF-IDF and cosine similarity"""
    
    def __init__(self, model_path: str = "keyword_model.pkl"):
        self.model_path = model_path
        self.vectorizer = None
        self.domain_corpus = {}
        self.keyword_embeddings = {}
        self.is_trained = False
        
        # Predefined domain vocabularies (keeps model small)
        self.domain_vocabularies = {
            'academic': [
                'research', 'study', 'analysis', 'methodology', 'literature', 'findings', 
                'experiment', 'data', 'results', 'conclusion', 'hypothesis', 'theory',
                'review', 'survey', 'evaluation', 'assessment', 'benchmark', 'performance',
                'algorithm', 'method', 'approach', 'technique', 'framework', 'model'
            ],
            'business': [
                'revenue', 'profit', 'growth', 'market', 'financial', 'investment',
                'strategy', 'analysis', 'performance', 'competitive', 'industry',
                'trends', 'forecast', 'metrics', 'kpi', 'roi', 'margin', 'share',
                'customer', 'product', 'service', 'operations', 'management'
            ],
            'medical': [
                'patient', 'treatment', 'diagnosis', 'therapy', 'clinical', 'medical',
                'health', 'disease', 'symptom', 'drug', 'medication', 'procedure',
                'outcome', 'efficacy', 'safety', 'trial', 'study', 'research'
            ],
            'technical': [
                'system', 'algorithm', 'implementation', 'architecture', 'design',
                'development', 'software', 'hardware', 'performance', 'optimization',
                'evaluation', 'testing', 'validation', 'framework', 'platform'
            ],
            'chemistry': [
                'reaction', 'compound', 'synthesis', 'mechanism', 'structure',
                'properties', 'analysis', 'molecular', 'chemical', 'organic',
                'kinetics', 'thermodynamics', 'catalyst', 'bond', 'electron'
            ]
        }
        
        self.load_or_create_model()
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data.get('vectorizer')
                    self.keyword_embeddings = model_data.get('embeddings', {})
                    self.is_trained = True
                print(f"✅ Loaded ML model from {self.model_path}")
            except Exception as e:
                print(f"⚠️  Error loading model: {e}. Creating new model...")
                self.create_lightweight_model()
        else:
            self.create_lightweight_model()
    
    def create_lightweight_model(self):
        """Create a lightweight TF-IDF based model"""
        print("🔧 Creating lightweight ML model for keyword expansion...")
        
        # Combine all domain vocabularies
        all_terms = []
        for domain, terms in self.domain_vocabularies.items():
            all_terms.extend(terms)
        
        # Create expanded corpus with related terms
        corpus = []
        for domain, terms in self.domain_vocabularies.items():
            # Create sentences with related terms
            for i in range(0, len(terms), 5):
                sentence = ' '.join(terms[i:i+5])
                corpus.append(sentence)
        
        # Train TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,  # Keep model small
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            
            # Create keyword embeddings
            feature_names = self.vectorizer.get_feature_names_out()
            for i, term in enumerate(feature_names):
                if i < tfidf_matrix.shape[1]:
                    self.keyword_embeddings[term] = tfidf_matrix[:, i].toarray().flatten()
            
            self.is_trained = True
            self.save_model()
            print(f"✅ Created model with {len(self.keyword_embeddings)} terms")
            
        except Exception as e:
            print(f"⚠️  Error creating model: {e}. Using rule-based fallback.")
            self.is_trained = False
    
    def save_model(self):
        """Save the trained model"""
        try:
            model_data = {
                'vectorizer': self.vectorizer,
                'embeddings': self.keyword_embeddings
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"💾 Model saved to {self.model_path}")
        except Exception as e:
            print(f"⚠️  Error saving model: {e}")
    
    def expand_keywords(self, keywords: List[str], max_expansions: int = 10) -> List[str]:
        """Expand keywords using ML similarity or rule-based fallback"""
        if not keywords:
            return keywords
        
        expanded = set(keywords)
        
        if self.is_trained and self.vectorizer and self.keyword_embeddings:
            expanded.update(self._ml_expand_keywords(keywords, max_expansions))
        else:
            expanded.update(self._rule_based_expand_keywords(keywords, max_expansions))
        
        return list(expanded)
    
    def _ml_expand_keywords(self, keywords: List[str], max_expansions: int) -> List[str]:
        """ML-based keyword expansion using TF-IDF similarity"""
        expanded = []
        
        try:
            # Get TF-IDF vectors for input keywords
            keyword_vectors = []
            valid_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in self.keyword_embeddings:
                    keyword_vectors.append(self.keyword_embeddings[keyword.lower()])
                    valid_keywords.append(keyword.lower())
            
            if not keyword_vectors:
                return self._rule_based_expand_keywords(keywords, max_expansions)
            
            # Calculate average keyword vector
            avg_vector = np.mean(keyword_vectors, axis=0)
            
            # Find similar terms
            similarities = {}
            for term, embedding in self.keyword_embeddings.items():
                if term not in valid_keywords:  # Don't include original keywords
                    similarity = cosine_similarity([avg_vector], [embedding])[0][0]
                    similarities[term] = similarity
            
            # Get top similar terms
            sorted_terms = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            expanded = [term for term, score in sorted_terms[:max_expansions] if score > 0.1]
            
        except Exception as e:
            print(f"⚠️  ML expansion failed: {e}. Using rule-based fallback.")
            expanded = self._rule_based_expand_keywords(keywords, max_expansions)
        
        return expanded
    
    def _rule_based_expand_keywords(self, keywords: List[str], max_expansions: int) -> List[str]:
        """Rule-based keyword expansion fallback"""
        expanded = []
        
        # Keyword association rules
        associations = {
            'research': ['study', 'analysis', 'investigation', 'methodology', 'findings'],
            'analysis': ['evaluation', 'assessment', 'examination', 'review', 'study'],
            'method': ['approach', 'technique', 'procedure', 'methodology', 'algorithm'],
            'result': ['outcome', 'finding', 'conclusion', 'output', 'achievement'],
            'data': ['information', 'dataset', 'statistics', 'metrics', 'evidence'],
            'performance': ['efficiency', 'effectiveness', 'results', 'benchmark', 'metrics'],
            'model': ['framework', 'system', 'algorithm', 'approach', 'architecture'],
            'financial': ['economic', 'monetary', 'fiscal', 'revenue', 'profit'],
            'market': ['industry', 'sector', 'business', 'commercial', 'trade'],
            'revenue': ['income', 'earnings', 'sales', 'profit', 'financial'],
            'student': ['learner', 'education', 'academic', 'study', 'learning'],
            'concept': ['idea', 'principle', 'theory', 'notion', 'understanding'],
            'chemistry': ['chemical', 'compound', 'reaction', 'molecular', 'organic'],
            'reaction': ['mechanism', 'process', 'synthesis', 'kinetics', 'pathway']
        }
        
        # Expand based on associations
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in associations:
                expanded.extend(associations[keyword_lower][:3])  # Take top 3 associations
        
        # Remove duplicates and limit
        expanded = list(set(expanded))[:max_expansions]
        
        return expanded
    
    def get_domain_specific_terms(self, text: str) -> List[str]:
        """Get domain-specific terms based on text content"""
        text_lower = text.lower()
        domain_scores = {}
        
        # Score each domain based on term frequency
        for domain, terms in self.domain_vocabularies.items():
            score = sum(1 for term in terms if term in text_lower)
            domain_scores[domain] = score
        
        # Get the dominant domain
        if domain_scores:
            dominant_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[dominant_domain] > 0:
                return self.domain_vocabularies[dominant_domain][:15]  # Return top 15 terms
        
        return []