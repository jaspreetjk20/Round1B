
import os
import pdfplumber
import json
import numpy as np
from collections import Counter, defaultdict
import joblib
import re

MODEL_PATH = "ml_model.joblib"

class HeadingOnlyPDFExtractor:
    def __init__(self):
        self.page_layouts = []
        self.global_stats = {}
        
    def analyze_document_structure(self, doc):
        """Analyze overall document structure before extraction using pdfplumber"""
        all_blocks = []
        page_stats = []
        
        for page_num, page in enumerate(doc.pages, start=1):
            page_width = page.width
            page_height = page.height
            
            # Extract text blocks from characters
            blocks_on_page = self.extract_text_blocks_from_chars(page, page_num)
            all_blocks.extend(blocks_on_page)
            
            # Analyze page layout
            page_stats.append({
                "page": page_num,
                "width": page_width,
                "height": page_height,
                "blocks": len(blocks_on_page),
                "is_multi_column": self.detect_multi_column(blocks_on_page, page_width)
            })
        
        self.page_layouts = page_stats
        
        # Global statistics
        sizes = [b["size"] for b in all_blocks]
        fonts = [b["font"] for b in all_blocks]
        
        self.global_stats = {
            "avg_size": np.mean(sizes) if sizes else 12,
            "median_size": np.median(sizes) if sizes else 12,
            "size_std": np.std(sizes) if sizes else 2,
            "common_fonts": Counter(fonts).most_common(5),
            "size_distribution": Counter(sizes).most_common(10),
            "body_text_size": self.identify_body_text_size(sizes)
        }
        
        return all_blocks
    
    def extract_text_blocks_from_chars(self, page, page_num):
        """Extract text blocks from pdfplumber characters, grouping by formatting"""
        chars = page.chars
        if not chars:
            return []
        
        blocks = []
        current_block = {
            "text": "",
            "font": "",
            "size": 0,
            "bbox": [float('inf'), float('inf'), 0, 0],  # [x0, y0, x1, y1]
            "page": page_num,
            "char_count": 0,
            "is_bold": False
        }
        
        # Sort characters by position (top to bottom, left to right)
        sorted_chars = sorted(chars, key=lambda c: (round(c['top'], 1), c['x0']))
        
        tolerance_y = 2  # Vertical tolerance for same line
        tolerance_x = 5  # Horizontal tolerance for continuity
        
        for i, char in enumerate(sorted_chars):
            char_font = char.get('fontname', '')
            char_size = round(char.get('size', 12), 1)
            char_text = char.get('text', '')
            
            # Skip whitespace-only characters for font/size comparison
            if char_text.strip() == '':
                # Add whitespace to current block if it exists
                if current_block["text"]:
                    current_block["text"] += char_text
                continue
            
            # Check if this character belongs to the current block
            # Same formatting and position continuity
            is_same_format = (
                char_font == current_block["font"] and 
                char_size == current_block["size"]
            )
            
            # Check position continuity (same line and close horizontally)
            is_continuous = False
            if current_block["char_count"] > 0:
                # Get the last character position from bbox
                last_char_y = (current_block["bbox"][1] + current_block["bbox"][3]) / 2
                current_char_y = (char['top'] + char['bottom']) / 2
                
                y_diff = abs(current_char_y - last_char_y)
                x_diff = char['x0'] - current_block["bbox"][2]
                
                is_continuous = (y_diff <= tolerance_y and x_diff <= tolerance_x)
            
            # Start new block if format changed or position discontinuous
            if (current_block["char_count"] > 0 and 
                (not is_same_format or not is_continuous)):
                
                # Finalize current block
                if current_block["text"].strip():
                    current_block["text"] = current_block["text"].strip()
                    current_block["is_bold"] = self.is_font_bold(current_block["font"])
                    blocks.append(current_block.copy())
                
                # Start new block
                current_block = {
                    "text": "",
                    "font": "",
                    "size": 0,
                    "bbox": [float('inf'), float('inf'), 0, 0],
                    "page": page_num,
                    "char_count": 0,
                    "is_bold": False
                }
            
            # Add character to current block
            current_block["text"] += char_text
            current_block["font"] = char_font
            current_block["size"] = char_size
            current_block["char_count"] += 1
            
            # Update bounding box
            current_block["bbox"][0] = min(current_block["bbox"][0], char['x0'])  # x0
            current_block["bbox"][1] = min(current_block["bbox"][1], char['top'])  # y0 (top)
            current_block["bbox"][2] = max(current_block["bbox"][2], char['x1'])  # x1
            current_block["bbox"][3] = max(current_block["bbox"][3], char['bottom'])  # y1 (bottom)
        
        # Don't forget the last block
        if current_block["char_count"] > 0 and current_block["text"].strip():
            current_block["text"] = current_block["text"].strip()
            current_block["is_bold"] = self.is_font_bold(current_block["font"])
            blocks.append(current_block)
        
        # Post-process to merge blocks that should be together but still detect inline bold
        merged_blocks = self.merge_nearby_blocks(blocks)
        
        # Add contextual information about inline bold text
        contextualized_blocks = self.add_inline_bold_context(merged_blocks)
        
        return contextualized_blocks
    
    def add_inline_bold_context(self, blocks):
        """Add context to detect if bold text is inline within a paragraph"""
        for i, block in enumerate(blocks):
            block["is_inline_bold"] = False
            
            if not block.get("is_bold", False):
                continue
            
            # Check if this bold block has non-bold text very close before or after
            y_center = (block["bbox"][1] + block["bbox"][3]) / 2
            
            for j, other_block in enumerate(blocks):
                if i == j or other_block.get("is_bold", False):
                    continue
                
                other_y_center = (other_block["bbox"][1] + other_block["bbox"][3]) / 2
                y_diff = abs(y_center - other_y_center)
                
                # If on same line (small y difference)
                if y_diff <= 5:
                    # Check horizontal proximity
                    # Either this block follows the other closely
                    if abs(block["bbox"][0] - other_block["bbox"][2]) <= 15:
                        block["is_inline_bold"] = True
                        break
                    # Or this block is followed by the other closely
                    elif abs(block["bbox"][2] - other_block["bbox"][0]) <= 15:
                        block["is_inline_bold"] = True
                        break
        
        return blocks
    
    def is_font_bold(self, font_name):
        """Check if font name indicates bold formatting"""
        if not font_name:
            return False
        return any(keyword in font_name.lower() for keyword in ["bold", "heavy", "black", "demi"])
    
    def merge_nearby_blocks(self, blocks):
        """Merge text blocks that are very close and likely part of same content"""
        if len(blocks) <= 1:
            return blocks
        
        merged = []
        current_group = [blocks[0]]
        
        for i in range(1, len(blocks)):
            prev_block = current_group[-1]
            curr_block = blocks[i]
            
            # Check if blocks should be merged
            should_merge = self.should_merge_blocks(prev_block, curr_block)
            
            if should_merge:
                current_group.append(curr_block)
            else:
                # Finalize current group
                if len(current_group) == 1:
                    merged.append(current_group[0])
                else:
                    merged_block = self.merge_block_group(current_group)
                    merged.append(merged_block)
                
                # Start new group
                current_group = [curr_block]
        
        # Handle last group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_block = self.merge_block_group(current_group)
            merged.append(merged_block)
        
        return merged
    
    def should_merge_blocks(self, block1, block2):
        """Determine if two blocks should be merged"""
        # Don't merge if significantly different sizes
        size_ratio = max(block1["size"], block2["size"]) / min(block1["size"], block2["size"])
        if size_ratio > 1.2:
            return False
        
        # Don't merge if very different formatting (one bold, one not)
        # BUT we want to be more lenient to keep text together unless it's clearly a heading
        if block1["is_bold"] != block2["is_bold"]:
            # Only separate if both blocks are reasonably long (potential headings vs body text)
            if len(block1["text"]) > 15 or len(block2["text"]) > 15:
                return False
        
        # Check vertical proximity (same line or very close)
        y1_center = (block1["bbox"][1] + block1["bbox"][3]) / 2
        y2_center = (block2["bbox"][1] + block2["bbox"][3]) / 2
        y_diff = abs(y1_center - y2_center)
        
        # Check horizontal proximity
        x_gap = block2["bbox"][0] - block1["bbox"][2]
        
        # Merge if on same line and reasonably close horizontally
        return y_diff <= 3 and 0 <= x_gap <= 20
    
    def merge_block_group(self, blocks):
        """Merge a group of blocks into a single block"""
        if len(blocks) == 1:
            return blocks[0]
        
        # Sort by horizontal position
        blocks.sort(key=lambda b: b["bbox"][0])
        
        merged = blocks[0].copy()
        merged["text"] = ""
        merged["char_count"] = 0
        
        # Merge text with spaces
        for i, block in enumerate(blocks):
            if i > 0:
                merged["text"] += " "
            merged["text"] += block["text"]
            merged["char_count"] += block["char_count"]
        
        # Update bounding box to encompass all blocks
        merged["bbox"] = [
            min(b["bbox"][0] for b in blocks),  # min x0
            min(b["bbox"][1] for b in blocks),  # min y0
            max(b["bbox"][2] for b in blocks),  # max x1
            max(b["bbox"][3] for b in blocks)   # max y1
        ]
        
        # Use the most common font and size, but preserve bold if any block is bold
        fonts = [b["font"] for b in blocks]
        sizes = [b["size"] for b in blocks]
        merged["font"] = Counter(fonts).most_common(1)[0][0]
        merged["size"] = Counter(sizes).most_common(1)[0][0]
        merged["is_bold"] = any(b["is_bold"] for b in blocks)
        
        return merged
    
    def identify_body_text_size(self, sizes):
        """Identify the most common font size (likely body text)"""
        if not sizes:
            return 12
        size_counts = Counter(sizes)
        return size_counts.most_common(1)[0][0]
    
    def detect_multi_column(self, blocks, page_width):
        """Detect if page has multi-column layout"""
        if len(blocks) < 10:
            return False
        
        # Group by approximate x-coordinates
        x_positions = [b["bbox"][0] for b in blocks]
        x_clusters = self.cluster_coordinates(x_positions, tolerance=50)
        
        # Multi-column if we have distinct x-coordinate clusters
        return len(x_clusters) >= 2
    
    def cluster_coordinates(self, coords, tolerance=50):
        """Cluster coordinates within tolerance"""
        coords = sorted(set(coords))
        clusters = []
        current_cluster = [coords[0]]
        
        for coord in coords[1:]:
            if coord - current_cluster[-1] <= tolerance:
                current_cluster.append(coord)
            else:
                clusters.append(current_cluster)
                current_cluster = [coord]
        
        clusters.append(current_cluster)
        return clusters
    
    def extract_blocks_advanced(self, pdf_path):
        """Advanced block extraction with strict heading focus"""
        with pdfplumber.open(pdf_path) as doc:
            all_raw_blocks = self.analyze_document_structure(doc)
        
        filtered_blocks = []
        
        for block in all_raw_blocks:
            # Strict filtering - only potential headings pass through
            if self.is_definitely_not_heading(block):
                continue
            
            # Enhanced feature extraction
            enhanced_block = self.enhance_block_features(block)
            filtered_blocks.append(enhanced_block)
        
        return filtered_blocks
    
    def enhance_block_features(self, block):
        """Add advanced features to block"""
        text = block["text"]
        font = block["font"]
        size = block["size"]
        bbox = block["bbox"]
        
        enhanced = block.copy()
        enhanced.update({
            "is_bold": block.get("is_bold", False),  # Use existing bold detection
            "is_italic": any(keyword in font.lower() for keyword in ["italic", "oblique"]),
            "is_upper": sum(c.isupper() for c in text) / len(text) if text else 0,
            "is_title_case": text.istitle(),
            "length": len(text),
            "word_count": len(text.split()),
            "has_numbers": bool(re.search(r'\d', text)),
            "starts_with_number": bool(re.match(r'^\d+[\.\)\s]', text)),
            "ends_with_colon": text.rstrip().endswith(':'),
            "is_all_caps": text.isupper() and len(text) > 2,
            "relative_size": size / self.global_stats["avg_size"],
            "size_zscore": (size - self.global_stats["avg_size"]) / max(self.global_stats["size_std"], 0.1),
            "size_vs_body": size / self.global_stats["body_text_size"],
            "x_position": bbox[0],
            "y_position": bbox[1],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
            "is_left_aligned": bbox[0] < 100,
            "font_family": self.extract_font_family(font),
            "is_common_font": font in [f[0] for f in self.global_stats["common_fonts"][:3]],
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "has_articles": bool(re.search(r'\b(the|a|an|this|that|these|those)\b', text, re.IGNORECASE)),
            "has_conjunctions": bool(re.search(r'\b(and|or|but|however|therefore|because|since|while|although)\b', text, re.IGNORECASE)),
            "is_inline_bold": block.get("is_inline_bold", False),  # Preserve context info
        })
        
        return enhanced
    
    def extract_font_family(self, font_name):
        """Extract base font family"""
        font_name = re.sub(r'[,\-](Bold|Italic|Regular|Light|Heavy|Black|Medium).*', '', font_name)
        return font_name.split(',')[0].split('-')[0].strip()
    
    def is_definitely_not_heading(self, block):
        """Strict filtering - exclude anything that's definitely not a heading"""
        text = block["text"].strip()
        size = block["size"]
        
        # Inline bold text is likely not a heading
        if block.get("is_inline_bold", False):
            return True
        
        # Empty or whitespace only
        if not text or len(text.strip()) < 2:
            return True
        
        # Too long to be a heading (strict limit)
        if len(text) > 120:
            return True
        
        # Multiple sentences are never headings
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings > 1:
            return True
        
        # Credit/course requirement patterns (common in syllabi/requirements)
        requirement_patterns = [
            r'^\d+\s+credits?\s+of\s+\w+',  # "4 credits of Math"
            r'^\d+\s+hours?\s+of\s+\w+',   # "3 hours of Science" 
            r'^\d+\s+units?\s+of\s+\w+',   # "2 units of English"
            r'^\d+\s+semesters?\s+of\s+\w+', # "2 semesters of Language"
            r'^\d+\s+(credit|hour|unit|semester|year)s?\s+(in|of|from)', # variations
            r'at least \d+.*course', # "at least one course"
            r'minimum.*\d+.*(credit|hour|unit)', # "minimum 3 credits"
            r'should be an? (AP|advanced|honors)', # "should be an AP Math class"
            r'must (include|contain|have)', # "must include one lab course"
            r'grade.*\d+.*or (higher|above)', # "grade of B or higher"
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in requirement_patterns):
            return True
        
        # Contains typical body text indicators
        body_indicators = [
            r'\b(said|says|according|reported|stated|mentioned|explained|described|noted|observed)\b',
            r'\b(however|therefore|moreover|furthermore|additionally|consequently|meanwhile|nevertheless)\b',
            r'\b(the|a|an)\s+(following|previous|next|above|below|first|second|third|last)\b',
            r'\b(in order to|as well as|such as|for example|for instance|in addition|in contrast)\b',
            r'\b(it is|there is|there are|this is|that is|these are|those are)\b',
            r'\b(can be|will be|has been|have been|was|were|are|is)\s+\w+ed\b',  # passive voice
            r'\([^)]*\)',  # parenthetical statements
            r'^\s*[-•·]\s+',  # bullet points
        ]
        
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in body_indicators):
            return True
        
        # Contains quotes or citations (typical in body text)
        if '"' in text or "'" in text or re.search(r'\[\d+\]|\(\d+\)', text):
            return True
        
        # Too small compared to body text
        if size < self.global_stats.get("body_text_size", 12) * 0.9:
            return True
        
        # Contains too many articles and prepositions (body text characteristic)
        article_count = len(re.findall(r'\b(the|a|an|this|that|these|those|in|on|at|by|for|with|from|to|of)\b', text, re.IGNORECASE))
        if len(text.split()) > 3 and article_count / len(text.split()) > 0.4:
            return True
        
        # Page numbers or references
        if re.match(r'^\s*\d+\s*$', text) or re.match(r'^(page|p\.)\s*\d+', text, re.IGNORECASE):
            return True
        
        # URLs, emails, or technical identifiers
        if re.search(r'(http|www\.|@|\.com|\.org|\.edu|\.gov)', text, re.IGNORECASE):
            return True
        
        return False
    
    def strict_heading_detection(self, blocks):
        """Ultra-strict heading detection with multiple validation layers"""
        if not blocks:
            return "", []
        
        heading_candidates = []
        
        for block in blocks:
            confidence = 0
            text = block["text"].strip()
            
            # PRIMARY INDICATORS (strong heading signals)
            
            # Size significantly larger than body text
            if block["size_vs_body"] >= 1.3:
                confidence += 5
            elif block["size_vs_body"] >= 1.15:
                confidence += 3
            
            # Bold formatting - but penalize if it appears to be inline bold
            if block["is_bold"]:
                if block.get("is_inline_bold", False):
                    # Significantly penalize inline bold text to prevent false positives
                    confidence -= 6  # This will counteract the bold bonus and add further penalty
                else:
                    confidence += 4  # Normal bonus for standalone bold text
            
            # Proper heading patterns
            heading_patterns = [
                r'^(Chapter|Section|Part|Appendix|Introduction|Conclusion|Summary|Abstract|Overview|Background|Methodology|Results|Discussion|References|Bibliography)\b',
                r'^\d+[\.\)]\s*[A-Z]',  # Numbered sections
                r'^[A-Z][A-Z\s]{3,25}$',  # Short all caps titles
                r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){0,4}$',  # Title case (max 5 words)
            ]
            
            if any(re.match(pattern, text) for pattern in heading_patterns):
                confidence += 4
            
            # STRUCTURAL INDICATORS
            
            # Short length (typical of headings)
            if 3 <= len(text) <= 50:
                confidence += 2
            elif 51 <= len(text) <= 80:
                confidence += 1
            else:
                confidence -= 2  # Too long or too short
            
            # Word count
            word_count = block["word_count"]
            if 1 <= word_count <= 8:
                confidence += 2
            elif word_count > 15:
                confidence -= 3
            
            # Ends with colon (section indicators)
            if block["ends_with_colon"]:
                confidence += 2
            
            # Starts with number (numbered sections)
            if block["starts_with_number"]:
                confidence += 3
            
            # Title case
            if block["is_title_case"] and word_count <= 10:
                confidence += 2
            
            # All caps (but not too long)
            if block["is_all_caps"] and len(text) <= 40:
                confidence += 2
            
            # NEGATIVE INDICATORS (reduce confidence)
            
            # Contains articles/conjunctions (body text indicators)
            if block["has_articles"] and word_count <= 5:
                confidence -= 2
            elif block["has_articles"] and word_count > 5:
                confidence -= 4
            
            if block["has_conjunctions"]:
                confidence -= 3
            
            # Multiple sentences
            if block["sentence_count"] > 1:
                confidence -= 5
            
            # Contains periods (except at the end)
            if text.count('.') > 1 or (text.count('.') == 1 and not text.endswith('.')):
                confidence -= 3
            
            # Too many common words
            common_words = len(re.findall(r'\b(the|a|an|and|or|but|in|on|at|by|for|with|from|to|of|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|can)\b', text, re.IGNORECASE))
            if common_words > word_count * 0.3:
                confidence -= 2
            
            # Very short bold text that's not a common heading pattern is suspicious
            if block["is_bold"] and len(text) < 10 and not any(re.match(pattern, text) for pattern in heading_patterns):
                confidence -= 3
            
            # FINAL VALIDATION
            
            # Must have minimum confidence to be considered
            if confidence >= 7:  # Very high threshold
                # Additional validation for edge cases
                if self.final_heading_validation(block):
                    heading_candidates.append((block, confidence))
        
        # Sort by confidence, then by size, then by position
        heading_candidates.sort(key=lambda x: (-x[1], -x[0]["size"], x[0]["page"], x[0]["y_position"]))
        
        # Remove duplicates and very similar headings
        filtered_candidates = self.remove_duplicate_headings(heading_candidates)
        
        # Assign levels
        outline = self.assign_heading_levels(filtered_candidates)
        title = outline[0]["text"] if outline else ""
        
        return title, outline
    
    def final_heading_validation(self, block):
        """Final validation to ensure this is truly a heading"""
        text = block["text"].strip()
        
        # Immediately reject inline bold text
        if block.get("is_inline_bold", False):
            return False
        
        # Must not be a complete sentence with subject and predicate
        if re.search(r'\b(is|are|was|were|has|have|had|will|would|could|should|may|might|can)\s+\w+', text, re.IGNORECASE):
            return False
        
        # Must not contain typical body text phrases
        body_phrases = [
            r'as shown in',
            r'according to',
            r'it should be noted',
            r'in this section',
            r'as described',
            r'for example',
            r'such as',
            r'in order to'
        ]
        
        if any(re.search(phrase, text, re.IGNORECASE) for phrase in body_phrases):
            return False
        
        # Must not be a caption
        if re.match(r'^(figure|table|chart|graph|image|photo|diagram)\s+\d+', text, re.IGNORECASE):
            return False
        
        # Additional check: if bold and very short, must be left-aligned
        if block["is_bold"] and block["word_count"] <= 2 and not block["is_left_aligned"]:
            return False
        
        return True
    
    def remove_duplicate_headings(self, candidates):
        """Remove duplicate or very similar headings"""
        if not candidates:
            return candidates
        
        filtered = []
        seen_texts = set()
        
        for block, confidence in candidates:
            text = block["text"].strip().lower()
            
            # Skip if we've seen this exact text
            if text in seen_texts:
                continue
            
            # Skip if very similar to existing heading
            is_duplicate = False
            for seen_text in seen_texts:
                if self.are_headings_similar(text, seen_text):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_texts.add(text)
                filtered.append((block, confidence))
        
        return filtered
    
    def are_headings_similar(self, text1, text2):
        """Check if two headings are too similar"""
        # Same length and high character overlap
        if abs(len(text1) - len(text2)) <= 2:
            common_chars = sum(1 for a, b in zip(text1, text2) if a == b)
            similarity = common_chars / max(len(text1), len(text2))
            if similarity > 0.8:
                return True
        
        # One is substring of another
        if text1 in text2 or text2 in text1:
            return True
        
        return False
    
    def assign_heading_levels(self, candidates):
        """Intelligently assign heading levels based on size and structure"""
        if not candidates:
            return []
        
        # Group by size
        size_groups = defaultdict(list)
        for block, conf in candidates:
            size_groups[block["size"]].append((block, conf))
        
        # Sort sizes descending
        sorted_sizes = sorted(size_groups.keys(), reverse=True)
        
        # Assign levels (max 3 levels)
        outline = []
        level_map = {}
        current_level = 1
        
        for size in sorted_sizes:
            if current_level <= 3:
                level_map[size] = f"H{current_level}"
                current_level += 1
            else:
                level_map[size] = "H3"
        
        # Create outline maintaining document order
        all_blocks = [item for size_list in size_groups.values() for item in size_list]
        all_blocks.sort(key=lambda x: (x[0]["page"], x[0]["y_position"]))
        
        for block, conf in all_blocks:
            outline.append({
                "level": level_map[block["size"]],
                "text": block["text"],
                "page": block["page"],
                "confidence": conf,
                "size": block["size"],
                "font": block["font"],
                "is_bold": block["is_bold"],
                "word_count": block["word_count"]
            })
        
        return outline
    
    def process(self, pdf_path):
        """Main processing function - extracts headings only"""
        blocks = self.extract_blocks_advanced(pdf_path)
        print(f"Extracted {len(blocks)} potential heading blocks from {pdf_path}")
        
        title, outline = self.strict_heading_detection(blocks)
        
        print(f"Detected {len(outline)} confirmed headings")
        return title, outline

def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = HeadingOnlyPDFExtractor()
    
    for file in os.listdir(input_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, file)
            print(f"\nProcessing {file}...")
            
            try:
                title, outline = extractor.process(pdf_path)
                
                output_data = {
                    "title": title,
                    "outline": outline,
                    "total_headings": len(outline),
                    "document_stats": extractor.global_stats,
                    "page_layouts": extractor.page_layouts
                }
                
                out_file = os.path.join(output_dir, file.replace(".pdf", ".json"))
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Saved {len(outline)} headings to {out_file}")
                
                # Print extracted headings for verification
                if outline:
                    print("\nExtracted Headings:")
                    for heading in outline:
                        print(f"  {heading['level']}: {heading['text']} (confidence: {heading['confidence']})")
                
            except Exception as e:
                print(f"❌ Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()
