"""
Website Pages Visualization App using OpenAI's text-embedding-3-small model
"""

import streamlit as st
import umap
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from io import StringIO
import requests
from urllib.parse import urljoin, urlparse
import re
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, List, Tuple
import json
from firecrawl import FirecrawlApp
import textstat
import time
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Firecrawl client
firecrawl = FirecrawlApp(api_key=os.getenv('FIRECRAWL_API_KEY'))

# Simplified quality weights based on Google's core signals
DEFAULT_QUALITY_WEIGHTS = {
    'content_quality': 0.4,    # Combined structure, readability, and technical depth
    'topical_depth': 0.3,     # How thoroughly the content covers its topic
    'site_focus': 0.2,        # Alignment with website's primary topic
    'freshness': 0.1          # Content recency
}

DEFAULT_CATEGORIES = {
    'crypto_news': {
        'weight': 0.95,
        'keywords': [
            'ข่าวคริปโต', 'cryptocurrency news', 'crypto news', 'ข่าวสาร', 
            'อัพเดท', 'update', 'ประกาศ', 'announcement'
        ]
    },
    'crypto_guides': {
        'weight': 0.90,
        'keywords': [
            'คู่มือ', 'วิธีการ', 'how to', 'guide', 'basic', 'พื้นฐาน',
            'เริ่มต้น', 'getting started', 'สอน', 'tutorial'
        ]
    },
    'crypto_reviews': {
        'weight': 0.85,
        'keywords': [
            'รีวิว', 'review', 'วิเคราะห์', 'analysis', 'ทดสอบ', 'test',
            'เปรียบเทียบ', 'compare', 'ข้อดี', 'ข้อเสีย'
        ]
    },
    'price_predictions': {
        'weight': 0.80,
        'keywords': [
            'ราคา', 'price', 'prediction', 'forecast', 'พยากรณ์', 'คาดการณ์',
            'แนวโน้ม', 'trend', 'technical analysis', 'วิเคราะห์ทางเทคนิค'
        ]
    },
    'top_recommendations': {
        'weight': 0.75,
        'keywords': [
            'อันดับ', 'ranking', 'top', 'best', 'แนะนำ', 'recommend',
            'ยอดนิยม', 'popular', 'น่าลงทุน', 'น่าสนใจ'
        ]
    }
}

# Default URLs for analysis
DEFAULT_URLS = [
    "https://cryptonews.com/th/cryptocurrency/next-crypto-to-explode/",
    "https://cryptonews.com/th/cryptocurrency/best-altcoins/",
    "https://cryptonews.com/th/cryptocurrency/best-utility-tokens/",
    "https://cryptonews.com/th/cryptocurrency/best-meme-coins/",
    "https://cryptonews.com/th/cryptocurrency/best-crypto-presales/",
    "https://cryptonews.com/th/cryptocurrency/best-crypto-icos/",
    "https://cryptonews.com/th/cryptocurrency/crypto-that-pays-dividends/",
    "https://cryptonews.com/th/cryptocurrency/best-crypto-under-1-dollar/",
    "https://cryptonews.com/th/cryptocurrency/fastest-growing-cryptocurrency/",
    "https://cryptonews.com/th/cryptocurrency/best-web3-coins/",
    "https://cryptonews.com/th/cryptocurrency/cheap-cryptocurrencies/",
    "https://cryptonews.com/th/cryptocurrency/best-crypto-portfolio-allocation/",
    "https://cryptonews.com/th/cryptocurrency/what-is-gamefi/",
    "https://cryptonews.com/th/cryptocurrency/the-most-popular-cryptocurrency-terms/"
]

class ContentQualityAnalyzer:
    def __init__(self, quality_weights=None, categories=None):
        self.model = "text-embedding-3-small"
        self.quality_weights = quality_weights or DEFAULT_QUALITY_WEIGHTS
        self.categories = categories or DEFAULT_CATEGORIES
        self.keywords = [keyword for category in self.categories.values() for keyword in category['keywords']]
    
    def chunk_text(self, text: str, max_tokens: int = 6000):
        """Split text into smaller chunks to fit within token limit."""
        # Use a smaller chunk size to ensure we stay well under the limit
        chunk_size = max_tokens // 2  # This gives us room for overhead
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Split text into sentences for more natural chunks
        sentences = text.split('. ')
        
        for sentence in sentences:
            # Rough estimate of tokens (4 chars ~= 1 token)
            sentence_tokens = len(sentence) // 4
            if current_size + sentence_tokens > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = []
                current_size = 0
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, handling long content by chunking."""
        try:
            # Clean and normalize text
            text = text.strip()
            if not text:
                return []
            
            # Use a more conservative token limit
            max_tokens = 6000  # text-embedding-3-small has 8192 limit, leave room for overhead
            
            # If text is too long, split into chunks
            if len(text) > max_tokens * 4:  # Rough estimate: 1 token ≈ 4 characters
                chunks = self.chunk_text(text, max_tokens)
                embeddings = []
                
                for chunk in chunks:
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                        
                    try:
                        response = client.embeddings.create(
                            input=chunk,
                            model=self.model
                        )
                        embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        st.warning(f"Skipping chunk due to error: {str(e)}")
                        continue
                
                # Average the embeddings if we have any
                if embeddings:
                    # Weighted average based on chunk length
                    weights = [len(chunk.strip()) for chunk in chunks if chunk.strip()]
                    total_weight = sum(weights)
                    weights = [w/total_weight for w in weights]
                    
                    # Calculate weighted average
                    avg_embedding = [
                        sum(w * e[i] for w, e in zip(weights, embeddings))
                        for i in range(len(embeddings[0]))
                    ]
                    return avg_embedding
                
                # If no embeddings were successful, try with just the beginning
                if chunks:
                    first_chunk = chunks[0].strip()
                    try:
                        response = client.embeddings.create(
                            input=first_chunk,
                            model=self.model
                        )
                        return response.data[0].embedding
                    except Exception as e:
                        st.error(f"Error getting embedding for first chunk: {str(e)}")
                        return []
            
            # For shorter text, get embedding directly
            response = client.embeddings.create(
                input=text,
                model=self.model
            )
            return response.data[0].embedding
            
        except Exception as e:
            st.error(f"Error in get_embedding: {str(e)}")
            return []

    def calculate_content_quality(self, content: str) -> float:
        """
        Calculate overall content quality score combining structure, readability, and technical depth.
        """
        try:
            # Extract text content
            text = content.get('markdown', '') if isinstance(content, dict) else str(content)
            if not text.strip():
                return 0.0
            
            # Structure analysis
            paragraphs = [p for p in text.split('\n\n') if p.strip()]
            has_headers = bool(re.findall(r'^#+ ', text, re.MULTILINE))
            has_lists = bool(re.findall(r'^\s*[-*+] ', text, re.MULTILINE))
            
            # Readability analysis
            readability_score = textstat.flesch_reading_ease(text) / 100
            
            # Length and depth analysis
            word_count = len(text.split())
            min_words = 300  # Minimum word count for a quality article
            max_words = 2500  # Optimal maximum word count
            length_score = min(word_count / min_words, max_words / word_count) if word_count > 0 else 0
            
            # Technical terms and concepts
            technical_terms = sum(1 for term in ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi', 'smart contract', 'token', 'mining', 'wallet', 'exchange'] if term.lower() in text.lower())
            tech_score = min(technical_terms / 10, 1.0)  # Normalize to 0-1
            
            # Calculate combined quality score
            structure_score = (0.3 * bool(paragraphs) +
                             0.3 * has_headers +
                             0.2 * has_lists +
                             0.2 * (len(paragraphs) >= 5))
            
            quality_score = (
                0.3 * structure_score +
                0.3 * readability_score +
                0.2 * length_score +
                0.2 * tech_score
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            st.error(f"Error calculating content quality: {str(e)}")
            return 0.5
    
    def calculate_topical_depth(self, content: str) -> float:
        """Calculate topical depth score based on content length and specific terms."""
        crypto_terms = [
            'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi',
            'smart contract', 'token', 'mining', 'wallet', 'exchange',
            'decentralized', 'consensus', 'protocol', 'network', 'transaction'
        ]
        
        # Normalize content
        content_lower = content.lower()
        
        # Calculate term frequency
        term_count = sum(content_lower.count(term) for term in crypto_terms)
        
        # Consider content length (assuming ideal length is 1500-2000 words)
        word_count = len(content.split())
        length_score = min(word_count / 2000.0, 1.0)
        
        # Combine metrics
        depth_score = (0.7 * min(term_count / 20.0, 1.0)) + (0.3 * length_score)
        return min(max(depth_score, 0.0), 1.0)

    def calculate_keyword_similarity(self, content: str, keywords: List[str]) -> float:
        """Calculate similarity between content and predefined keywords."""
        try:
            if not content.strip():
                return 0.0
                
            # Get content embedding
            content_embedding = self.get_embedding(content)
            if not content_embedding:
                st.warning("Could not generate content embedding, using default score")
                return 0.5
                
            # Get keyword embeddings (using cached if available)
            if not hasattr(self, '_keyword_embeddings'):
                self._keyword_embeddings = []
                for keyword in self.keywords:
                    emb = self.get_embedding(keyword)
                    if emb:  # Only add valid embeddings
                        self._keyword_embeddings.append(emb)
            
            if not self._keyword_embeddings:
                st.warning("No valid keyword embeddings available")
                return 0.5
            
            # Calculate similarities
            similarities = []
            for keyword_embedding in self._keyword_embeddings:
                try:
                    similarity = cosine_similarity(
                        np.array(content_embedding).reshape(1, -1),
                        np.array(keyword_embedding).reshape(1, -1)
                    )[0][0]
                    similarities.append(similarity)
                except Exception as e:
                    st.warning(f"Error calculating similarity: {str(e)}")
                    continue
            
            # Return average similarity if we have any valid calculations
            return float(np.mean(similarities)) if similarities else 0.5
            
        except Exception as e:
            st.error(f"Error calculating keyword similarity: {str(e)}")
            return 0.5
    
    def calculate_freshness_score(self, publish_date: datetime) -> float:
        """Calculate content freshness score using a decay model."""
        if not publish_date:
            return 0.5  # Default score for unknown dates
        
        now = datetime.now()
        
        age_days = (now - publish_date).days
        
        # Exponential decay model with half-life of 90 days
        half_life = 90
        decay_constant = np.log(2) / half_life
        freshness_score = np.exp(-decay_constant * age_days)
        
        return float(freshness_score)
    
    def calculate_site_focus_score(self, content: str, category: str) -> float:
        """Calculate how well the content aligns with site's crypto focus."""
        category_weights = {
            'crypto_news': 0.95,
            'crypto_guides': 0.90,
            'crypto_reviews': 0.85,
            'price_predictions': 0.80,
            'top_recommendations': 0.75
        }
        
        base_score = category_weights.get(category, 0.5)
        
        # Adjust based on crypto-specific content
        crypto_terms = [
            'blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'defi',
            'smart contract', 'token', 'mining', 'wallet', 'exchange'
        ]
        
        term_presence = sum(term in content.lower() for term in crypto_terms) / len(crypto_terms)
        
        # Combine scores with weights
        return (0.7 * base_score) + (0.3 * term_presence)

    def calculate_overall_score(self, content: Dict) -> Dict[str, float]:
        """Calculate overall content score using simplified weights."""
        try:
            # Get individual scores
            quality_score = self.calculate_content_quality(content.get('content', ''))
            topical_score = self.calculate_topical_depth(content.get('content', ''))
            focus_score = self.calculate_site_focus_score(content.get('content', ''), 'crypto_guides')
            freshness_score = self.calculate_freshness_score(
                content.get('publish_date') if isinstance(content.get('publish_date'), datetime) 
                else None
            )
            
            # Calculate weighted score
            weighted_score = (
                quality_score * self.quality_weights['content_quality'] +
                topical_score * self.quality_weights['topical_depth'] +
                focus_score * self.quality_weights['site_focus'] +
                freshness_score * self.quality_weights['freshness']
            )
            
            return {
                'Overall Score': round(weighted_score, 3),
                'Content Quality': round(quality_score, 3),
                'Topical Depth': round(topical_score, 3),
                'Site Focus': round(focus_score, 3),
                'Freshness': round(freshness_score, 3)
            }
            
        except Exception as e:
            st.error(f"Error calculating overall score: {str(e)}")
            return None

def detect_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in the dataset using the Interquartile Range (IQR) method.
    
    Args:
        df: DataFrame containing the metrics
        columns: List of column names to check for outliers
        threshold: IQR multiplier for outlier detection (default: 1.5)
    
    Returns:
        DataFrame with outlier flags
    """
    outlier_flags = pd.DataFrame(index=df.index)
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_flags[f'{col}_outlier'] = (
                (df[col] < lower_bound) | (df[col] > upper_bound)
            )
    
    return outlier_flags

def create_visualization(analyzer, urls_data, embeddings_list):
    """Create UMAP visualization of content embeddings."""
    try:
        if not embeddings_list or not urls_data:
            st.warning("No valid embeddings or data available for visualization")
            return None
        
        # Filter out any empty embeddings and ensure all have the same shape
        valid_embeddings = []
        valid_urls = []
        
        for emb, url_data in zip(embeddings_list, urls_data):
            if isinstance(emb, list) and len(emb) == 1536:  # Check for correct embedding size
                valid_embeddings.append(emb)
                valid_urls.append(url_data)
        
        if not valid_embeddings:
            st.warning("No valid embeddings found for visualization")
            return None
        
        # Convert to numpy array
        embeddings_array = np.array(valid_embeddings)
        
        # Extract categories from URLs
        categories = []
        for url in valid_urls:
            # Extract category from URL path
            path_parts = url['url'].split('/')
            category = 'other'
            for part in path_parts:
                if part in ['cryptocurrency', 'price-predictions', 'news']:
                    category = part
                    break
            categories.append(category)
        
        # Create DataFrame with embeddings and metadata
        df = pd.DataFrame({
            'url': [url['url'] for url in valid_urls],
            'title': [url.get('title', '') for url in valid_urls],
            'category': categories,
            'quality_score': [analyzer.calculate_content_quality(url) for url in valid_urls]
        })
        
        # Create UMAP visualization
        reducer = umap.UMAP(random_state=42)
        
        if len(embeddings_array) > 0:
            umap_embeddings = reducer.fit_transform(embeddings_array)
            
            # Create scatter plot
            fig = px.scatter(
                x=umap_embeddings[:, 0],
                y=umap_embeddings[:, 1],
                color=df['category'],
                hover_data={
                    'URL': df['url'],
                    'Title': df['title'],
                    'Quality Score': df['quality_score']
                },
                title=f'Content Embedding Visualization ({len(valid_embeddings)} valid articles)'
            )
            
            return fig
        
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.error(f"Embeddings shape: {np.array(embeddings_list).shape if embeddings_list else 'No embeddings'}")
        return None

def scrape_cryptonews_content(url: str) -> Dict:
    """
    Scrape content from a Cryptonews.com/th URL using Firecrawl with improved reliability.
    
    Args:
        url (str): The URL to scrape
        
    Returns:
        Dict: Dictionary containing scraped content and metadata
    """
    try:
        st.write(f"Attempting to scrape: {url}")
        
        # Initialize parameters for scraping
        params = {
            'url': url,
            'wait': 5,  # Increase wait time
            'js': True,
            'headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        # First attempt to scrape
        st.write("Sending request to Firecrawl...")
        result = firecrawl.scrape_url(params)
        
        # Check if we got a crawl_id
        if hasattr(result, 'crawl_id'):
            crawl_id = result.crawl_id
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                # Check crawl status
                status = firecrawl.check_crawl_status(crawl_id)
                st.write(f"Crawl status: {status}")
                
                if status.get('status') == 'completed':
                    result = firecrawl.get_crawl_result(crawl_id)
                    break
                elif status.get('status') == 'failed':
                    st.error(f"Crawl failed: {status.get('error')}")
                    break
                
                time.sleep(2)  # Wait before checking again
                retry_count += 1
        
        st.write("Firecrawl response received:", bool(result))
        
        if not result or not hasattr(result, 'markdown'):
            st.write(f"No data in result for {url}")
            st.write("Result structure:\n")
            st.write(json.dumps(result.__dict__ if hasattr(result, '__dict__') else result, indent=2))
            return generate_mock_crypto_content(url)
        
        # Extract content
        content = result.markdown
        
        # Basic content validation
        if not content or len(content.strip()) < 100:  # Minimum content length
            st.warning(f"Content too short or empty for {url}, falling back to mock data")
            return generate_mock_crypto_content(url)
        
        # Extract metadata
        meta_description = result.meta_description if hasattr(result, 'meta_description') else ""
        title = result.title if hasattr(result, 'title') else ""
        
        # Try to extract date from content or URL
        content_date = extract_date_from_content(content, url)
        
        return {
            'url': url,
            'title': title,
            'content': content,
            'meta_description': meta_description,
            'content_date': content_date
        }
        
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        st.write(f"Full error details: {str(e)}")
        return generate_mock_crypto_content(url)

def batch_scrape_urls(urls: List[str], max_urls: int = 10) -> List[Dict]:
    """
    Batch scrape multiple URLs using Firecrawl's scraping endpoint.
    Limited to max_urls during testing phase.
    """
    try:
        # Limit number of URLs during testing
        urls = urls[:max_urls]
        st.write(f"Batch scraping {len(urls)} URLs...")
        
        # Simplest possible parameters
        params = {
            'formats': ['markdown']
        }
        
        # Process URLs sequentially with rate limiting
        results = []
        for url in urls:
            try:
                st.write(f"Processing URL: {url}")
                
                # Make the request using scrape_url
                result = firecrawl.scrape_url(url, params)
                
                if result and isinstance(result, dict) and (result.get('markdown') or result.get('data', {}).get('markdown')):
                    # Try both possible response structures
                    content = result.get('markdown') or result.get('data', {}).get('markdown', '')
                    
                    if content:
                        st.success(f"Successfully scraped: {url}")
                        
                        # Extract title from content if possible
                        title = ''
                        lines = content.split('\n')
                        if lines and lines[0].startswith('# '):
                            title = lines[0].replace('# ', '')
                        
                        results.append({
                            'url': url,
                            'markdown': content,
                            'title': title,
                            'meta_description': '',  # We'll skip metadata for now
                            'publish_date': None
                        })
                    else:
                        st.warning(f"No content found for {url}, using mock data")
                        results.append(generate_mock_crypto_content(url))
                else:
                    st.warning(f"Invalid response for {url}, using mock data")
                    results.append(generate_mock_crypto_content(url))
                
            except Exception as e:
                st.error(f"Error scraping {url}: {str(e)}")
                results.append(generate_mock_crypto_content(url))
            
            time.sleep(2)  # Rate limiting between requests
        
        return process_batch_results(results, urls)
        
    except Exception as e:
        st.error(f"Error in batch scraping: {str(e)}")
        return [generate_mock_crypto_content(url) for url in urls]

def process_batch_results(results: List[Dict], original_urls: List[str]) -> List[Dict]:
    """Process the results from batch scraping."""
    processed_results = []
    
    for result, url in zip(results, original_urls):
        try:
            content = result.get('markdown', '')
            if isinstance(content, str) and len(content.strip()) >= 100:
                # Extract metadata
                meta_description = result.get('meta_description', '')
                title = result.get('title', '')
                
                # Try to get publish date from result or extract from content
                content_date = None
                if result.get('publish_date'):
                    try:
                        content_date = datetime.fromisoformat(result['publish_date'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        content_date = extract_date_from_content(content, url)
                else:
                    content_date = extract_date_from_content(content, url)
                
                processed_results.append({
                    'url': url,
                    'title': title,
                    'content': content,
                    'meta_description': meta_description,
                    'content_date': content_date
                })
            else:
                st.warning(f"Invalid or short content for {url}, using mock data")
                processed_results.append(generate_mock_crypto_content(url))
            
        except Exception as e:
            st.error(f"Error processing result for {url}: {str(e)}")
            processed_results.append(generate_mock_crypto_content(url))
    
    return processed_results

def extract_date_from_content(content: str, url: str) -> datetime:
    """Extract publication date from content or URL."""
    try:
        # Common date patterns in Thai content
        patterns = [
            r'(\d{1,2})\s*(วัน|ชั่วโมง|นาที)ที่ผ่านมา',
            r'(\d{1,2})\s*(ชั่วโมง|นาที)ที่แล้ว',
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{4})-(\d{2})-(\d{2})',
        ]
        
        now = datetime.now()
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                match = matches[0]
                if isinstance(match, tuple):
                    if 'ที่ผ่านมา' in pattern or 'ที่แล้ว' in pattern:
                        value = int(match[0])
                        unit = match[1]
                        if 'วัน' in unit:
                            return now - timedelta(days=value)
                        elif 'ชั่วโมง' in unit:
                            return now - timedelta(hours=value)
                        elif 'นาที' in unit:
                            return now - timedelta(minutes=value)
                    else:
                        # Handle standard date formats
                        try:
                            if len(match) == 3:
                                day, month, year = map(int, match)
                                return datetime(year, month, day)
                        except ValueError:
                            continue
        
        # Try to extract date from URL
        url_match = re.search(r'/(\d{4})/(\d{2})/', url)
        if url_match:
            year, month = map(int, url_match.groups())
            return datetime(year, month, 1)
        
        return now
        
    except Exception as e:
        st.warning(f"Error extracting date: {str(e)}")
        return datetime.now()

def generate_mock_crypto_content(url: str) -> Dict:
    """Generate mock content for crypto news URLs with enhanced quality metrics."""
    categories = {
        'crypto_news': {
            'titles': [
                "ข่าวคริปโตล่าสุด",
                "Cryptocurrency News Today",
                "Crypto Market Updates"
            ],
            'descriptions': [
                "ข่าวคริปโตล่าสุดและอัปเดตตลาด",
                "Cryptocurrency news and market updates",
                "Stay up-to-date with the latest crypto news"
            ]
        },
        'crypto_guides': {
            'titles': [
                "คู่มือคริปโตสำหรับผู้เริ่มต้น",
                "Cryptocurrency Guide for Beginners",
                "Getting Started with Crypto"
            ],
            'descriptions': [
                "คู่มือคริปโตสำหรับผู้เริ่มต้น",
                "A comprehensive guide to cryptocurrency for beginners",
                "Learn the basics of cryptocurrency and how to get started"
            ],
            'alignment_score': 0.90  # Strong alignment with crypto focus
        },
        'crypto_reviews': {
            'titles': [
                "รีวิวคริปโต",
                "Cryptocurrency Reviews",
                "Crypto Reviews and Ratings"
            ],
            'descriptions': [
                "รีวิวคริปโตและอันดับ",
                "In-depth reviews and ratings of cryptocurrencies",
                "Find the best cryptocurrencies with our expert reviews"
            ],
            'alignment_score': 0.85  # Good alignment with crypto focus
        },
        'price_predictions': {
            'titles': [
                "การคาดการณ์ราคาคริปโต",
                "Cryptocurrency Price Predictions",
                "Crypto Price Forecast"
            ],
            'descriptions': [
                "การคาดการณ์ราคาคริปโต",
                "Accurate cryptocurrency price predictions and forecasts",
                "Stay ahead of the market with our crypto price predictions"
            ],
            'alignment_score': 0.80  # Moderate alignment with crypto focus
        },
        'top_recommendations': {
            'titles': [
                "อันดับคริปโต",
                "Top Cryptocurrencies",
                "Best Cryptocurrencies to Invest"
            ],
            'descriptions': [
                "อันดับคริปโต",
                "Top-rated cryptocurrencies and investment opportunities",
                "Find the best cryptocurrencies to invest in with our expert recommendations"
            ],
            'alignment_score': 0.75  # Lower alignment due to general finance focus
        }
    }
    
    # Determine category based on URL
    if 'blockchain' in url.lower() or 'fundamentals' in url.lower():
        category = 'crypto_guides'
    elif 'price' in url.lower() or 'market' in url.lower():
        category = 'price_predictions'
    elif 'defi' in url.lower() or 'protocol' in url.lower():
        category = 'crypto_reviews'
    elif 'news' in url.lower() or 'update' in url.lower():
        category = 'crypto_news'
    elif 'investment' in url.lower() or 'portfolio' in url.lower():
        category = 'top_recommendations'
    else:
        category = 'crypto_guides'
    
    # Generate content
    cat_data = categories[category]
    title = random.choice(cat_data['titles'])
    desc = random.choice(cat_data['descriptions'])
    
    # Add a mock date to the content for testing
    current_date = datetime.now()
    random_days_ago = random.randint(0, 365)  # Random date within the last year
    content_date = current_date - timedelta(days=random_days_ago)
    content = f"{title}. {desc}\n\nPublished: {content_date.strftime('%B %d, %Y')}"
    
    # Calculate quality metrics
    analyzer = ContentQualityAnalyzer()
    topical_depth = analyzer.calculate_topical_depth(content)
    content_quality = analyzer.calculate_content_quality(content)
    site_focus_score = analyzer.calculate_site_focus_score(content, category)
    freshness_score = analyzer.calculate_freshness_score(content_date)
    
    # Calculate overall quality score
    quality_score = (
        DEFAULT_QUALITY_WEIGHTS['topical_depth'] * topical_depth +
        DEFAULT_QUALITY_WEIGHTS['content_quality'] * content_quality +
        DEFAULT_QUALITY_WEIGHTS['site_focus'] * site_focus_score +
        DEFAULT_QUALITY_WEIGHTS['freshness'] * freshness_score
    )
    
    # Generate embedding using OpenAI's model
    embedding = analyzer.get_embedding(content)
    
    return {
        'url': url,
        'title': title,
        'meta_description': desc,
        'content': content,
        'category': category,
        'embedding': embedding,
        'topical_depth': topical_depth,
        'content_quality': content_quality,
        'site_focus_score': site_focus_score,
        'freshness_score': freshness_score,
        'publish_date': content_date,
        'quality_score': quality_score
    }

def generate_dummy_embeddings(n_samples, n_dimensions=1536):
    """Generate dummy embeddings for testing."""
    return np.random.normal(0, 1, (n_samples, n_dimensions))

@st.cache_data
def process_urls(urls, _use_mock=False):
    """Process list of URLs and generate content."""
    if _use_mock:
        pages_data = [generate_mock_crypto_content(url) for url in urls]
    else:
        # Use batch scraping for better performance
        pages_data = batch_scrape_urls(urls)
    
    # Create analyzer instance inside the function
    analyzer = ContentQualityAnalyzer()
    
    processed_pages = []
    for page in pages_data:
        if not page:
            continue
        
        content = page['content']
        url = page['url']
        
        try:
            st.write(f"Processing URL: {url}")
            
            # Get embedding
            embedding = analyzer.get_embedding(content)
            
            # Calculate metrics
            content_quality = analyzer.calculate_content_quality(content)
            topical_depth = analyzer.calculate_topical_depth(content)
            site_focus_score = analyzer.calculate_site_focus_score(content, 'crypto_guides')
            
            # Update page data with calculated metrics
            page.update({
                'embedding': embedding,
                'quality_score': content_quality,
                'topical_depth': topical_depth,
                'content_quality': content_quality,
                'site_focus_score': site_focus_score,
                'freshness_score': analyzer.calculate_freshness_score(page.get('content_date'))
            })
            processed_pages.append(page)
            
        except Exception as e:
            st.error(f"Error processing {url}: {str(e)}")
            continue
    
    return processed_pages, [p['embedding'] for p in processed_pages if 'embedding' in p]

def main():
    """Main function to run the Streamlit app."""
    st.title("Content Quality & Topic Alignment Analyzer")
    st.write("Analyze content quality and topic alignment using OpenAI's text-embedding-3-small model")
    
    use_default = st.checkbox("Use default cryptocurrency articles", value=True)
    
    if use_default:
        urls = DEFAULT_URLS
        st.info(f"Using {len(urls)} default cryptocurrency articles")
    else:
        urls_text = st.text_area("Enter URLs (one per line)")
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    if st.button("Analyze Content"):
        if urls:
            st.write("Processing webpage contents...")
            
            # Process URLs and get embeddings - removed analyzer parameter
            pages_data, embeddings = process_urls(urls)
            
            if pages_data and embeddings:
                # Create visualization with a new analyzer instance
                analyzer = ContentQualityAnalyzer()
                fig = create_visualization(analyzer, pages_data, embeddings)
                
                if fig:
                    # Display visualization
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display simplified metrics table
                    st.subheader("Content Quality Metrics")
                    
                    # Create metrics DataFrame
                    metrics_data = []
                    for page in pages_data:
                        metrics = analyzer.calculate_overall_score(page)
                        metrics['URL'] = page['url']
                        metrics['Title'] = page.get('title', '')
                        metrics_data.append(metrics)
                    
                    if metrics_data:
                        metrics_df = pd.DataFrame(metrics_data)
                        # Round numeric columns to 2 decimal places
                        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
                        metrics_df[numeric_cols] = metrics_df[numeric_cols].round(2)
                        
                        # Reorder columns to put URL and Title first
                        cols = ['URL', 'Title'] + [col for col in metrics_df.columns if col not in ['URL', 'Title']]
                        metrics_df = metrics_df[cols]
                        
                        st.dataframe(metrics_df, use_container_width=True)
                else:
                    st.error("Could not create visualization. Please check the error messages above.")
            else:
                st.error("No valid content or embeddings found for analysis.")
        else:
            st.error("Please enter at least one valid URL.")

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Content Quality & Topic Alignment Analyzer",
        page_icon="🎯",
        layout="wide"
    )

    # Title and description
    st.title("Content Quality & Topic Alignment Analyzer")
    st.markdown("""
    This app analyzes your website's content quality and topic alignment using OpenAI's text-embedding-3-small model.
    It evaluates content based on multiple factors and provides detailed quality metrics.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Category Configuration
    st.sidebar.subheader("Content Categories")
    show_category_editor = st.sidebar.checkbox("Edit Categories", False)
    
    if show_category_editor:
        categories = {}
        for category, data in DEFAULT_CATEGORIES.items():
            st.sidebar.markdown(f"**{category.replace('_', ' ').title()}**")
            weight = st.sidebar.slider(
                f"{category} weight",
                0.0, 1.0, data['weight'],
                key=f"cat_weight_{category}"
            )
            keywords = st.sidebar.text_area(
                f"{category} keywords",
                ", ".join(data['keywords']),
                key=f"cat_keywords_{category}"
            )
            categories[category] = {
                'weight': weight,
                'keywords': [k.strip() for k in keywords.split(",")]
            }
    else:
        categories = DEFAULT_CATEGORIES
    
    # Run main app
    # Use default weights directly, no UI configuration needed
    analyzer = ContentQualityAnalyzer(categories=categories)
    main()
