# keyword_embedding_app.py

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import urllib3
import json
from openai import OpenAI
import httpx
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Create OpenAI client for embeddings with custom transport
client = httpx.Client(verify=False)
openai_client = OpenAI(
    api_key=Config.OPENAI_API_KEY,
    http_client=client
)

# Create a pool manager with retry logic for OpenRouter
http = urllib3.PoolManager(
    retries=urllib3.Retry(3, backoff_factor=1)
)

def get_embedding_cached(text, model="text-embedding-3-small"):
    """
    Get embedding for a single text using OpenAI API directly.
    """
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding for text: {text}\n{e}")
        return None

def process_text(text):
    """
    Convert text to a cleaner format.
    """
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.title()

def call_openai_api(prompt, temperature=0.7, max_tokens=150):
    """
    Call the OpenAI Chat API through OpenRouter to summarize clusters.
    """
    try:
        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://github.com/codebase",
            "X-Title": "Embedding-Visualization",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek/deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = http.request(
            'POST',
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            body=json.dumps(payload).encode('utf-8')
        )
        
        if response.status != 200:
            error_msg = response.data.decode('utf-8')
            st.write("Debug - Error Response:", error_msg)
            raise Exception(f"API request failed with status {response.status}: {error_msg}")
            
        result = json.loads(response.data.decode('utf-8'))
        summary = result["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        st.error(f"Error calling the LLM: {e}")
        return ""

def shorten_text(text, max_length=20):
    """
    Shorten text for labels while preserving meaning.
    """
    text = text.strip()
    if len(text) <= max_length:
        return text
    words = text.split()
    if len(words) <= 2:
        return text[:max_length] + "..."
    return " ".join(words[:2]) + "..."

def merge_small_clusters(embeddings_array, labels, min_size=3):
    """
    Merge clusters that are smaller than min_size into the nearest large cluster.
    """
    unique_labels = np.unique(labels)
    cluster_sizes = np.array([np.sum(labels == label) for label in unique_labels])
    
    # Find small clusters
    small_clusters = unique_labels[cluster_sizes < min_size]
    large_clusters = unique_labels[cluster_sizes >= min_size]
    
    if len(large_clusters) == 0:
        # If no large clusters exist, keep the original labels
        return labels
    
    # For each small cluster
    for small_cluster in small_clusters:
        # Get indices of points in small cluster
        cluster_indices = np.where(labels == small_cluster)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Calculate mean distance to each large cluster
        distances = []
        for large_cluster in large_clusters:
            large_cluster_indices = np.where(labels == large_cluster)[0]
            cluster_distances = cosine_distances(
                embeddings_array[cluster_indices],
                embeddings_array[large_cluster_indices]
            ).mean()
            distances.append(cluster_distances)
        
        # Find closest large cluster
        closest_cluster = large_clusters[np.argmin(distances)]
        
        # Merge small cluster into closest large cluster
        labels[cluster_indices] = closest_cluster
    
    return labels

def get_cluster_terms(texts, labels, cluster_id):
    """
    Get the most representative terms for a cluster using TF-IDF.
    """
    # Create TF-IDF vectorizer
    from pythainlp.tokenize import word_tokenize

    def thai_tokenizer(text):
        return word_tokenize(text)

    vectorizer = TfidfVectorizer(tokenizer=thai_tokenizer)
    
    # Get texts for this cluster
    cluster_texts = [text for text, label in zip(texts, labels) if label == cluster_id]
    if not cluster_texts:
        return []
    
    # Calculate TF-IDF for cluster texts
    cluster_tfidf = vectorizer.fit_transform([' '.join(cluster_texts)])
    
    # Get feature names (terms)
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Get top terms indices
    top_terms_indices = np.argsort(cluster_tfidf.toarray().flatten())[-10:]  # Get more terms
    
    # Return top terms
    return feature_names[top_terms_indices][::-1]

def format_cluster_terms(terms, max_per_line=5):
    """Format terms in multiple lines for better readability"""
    lines = []
    for i in range(0, len(terms), max_per_line):
        line_terms = terms[i:i + max_per_line]
        lines.append(", ".join(line_terms))
    return "\n".join(lines)

def create_visualization(texts, max_length=20, n_clusters=5, min_cluster_size=3):
    """
    Create a visualization of text clusters.
    """
    if not texts:
        st.warning("No texts to visualize.")
        return

    # Get embeddings for all texts
    embeddings = []
    processed_texts = []
    
    with st.spinner('Getting embeddings...'):
        for text in texts:
            embedding = get_embedding_cached(text)
            if embedding is not None:
                embeddings.append(embedding)
                processed_texts.append(text)
    
    if not embeddings:
        st.warning("Could not get embeddings for any texts.")
        return
    
    # Convert embeddings to numpy array
    X = np.array(embeddings)
    
    # Perform clustering
    with st.spinner('Clustering...'):
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)
        
        # Merge small clusters
        labels = merge_small_clusters(X, labels, min_size=min_cluster_size)
    
    # Reduce dimensionality for visualization
    with st.spinner('Reducing dimensionality...'):
        tsne = TSNE(n_components=2, random_state=42)
        X_2d = tsne.fit_transform(X)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Plot points for each cluster
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_texts = [shorten_text(text, max_length) for text, is_in_cluster in zip(processed_texts, mask) if is_in_cluster]
        
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0],
            y=X_2d[mask, 1],
            mode='markers+text',
            name=f'Cluster {cluster_id}',
            text=cluster_texts,
            hovertext=cluster_texts,
            hoverinfo='text',
            textposition='top center'
        ))
    
    # Update layout
    fig.update_layout(
        title='Text Clusters Visualization',
        xaxis_title='t-SNE dimension 1',
        yaxis_title='t-SNE dimension 2',
        showlegend=True,
        hovermode='closest',
        height=600  # Fixed height to keep it above the fold
    )
    
    # Show plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Display cluster analysis in columns
    st.markdown("## Cluster Analysis")
    
    # Calculate number of columns (2 or 3 depending on number of clusters)
    num_cols = min(3, len(unique_labels))
    cols = st.columns(num_cols)
    
    # Generate cluster summaries
    with st.spinner('Generating cluster summaries...'):
        for idx, cluster_id in enumerate(unique_labels):
            col_idx = idx % num_cols
            cluster_texts = [text for text, label in zip(processed_texts, labels) if label == cluster_id]
            
            if cluster_texts:
                with cols[col_idx]:
                    # Debug info (collapsed by default)
                    with st.expander("🔍 Debug Info", expanded=False):
                        st.text(f"API Key present: {bool(Config.OPENROUTER_API_KEY)}")
                        st.text("Response Status: 200")
                    
                    # Cluster header
                    st.markdown(f"#### Cluster {cluster_id}")
                    
                    # Analysis section
                    st.markdown("**Analysis:**")
                    prompt = (
                        "Analyze these related texts and provide:\n"
                        "1. Main Theme (1 sentence)\n"
                        "2. Key Points (2-3 bullet points)\n\n"
                        f"Texts: {' | '.join(cluster_texts)}"
                    )
                    analysis = call_openai_api(prompt)
                    st.markdown(analysis)
                    
                    # Texts section
                    st.markdown("**Texts:**")
                    for text in cluster_texts:
                        st.text(text)
                    
                    # Add some spacing between clusters
                    st.markdown("---")

def main():
    st.set_page_config(page_title="Keyword Embedding Visualisation and Clustering", layout="wide")
    
    # Initialize session state
    if 'texts' not in st.session_state:
        st.session_state.texts = []
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = []
    if 'labels' not in st.session_state:
        st.session_state.labels = []
    if 'X_2d' not in st.session_state:
        st.session_state.X_2d = None
    
    st.title("Keyword Embedding Visualization and Clustering")
    
    # Text input area
    text_input = st.text_area(
        "Enter keywords (one per line)",
        height=150,
        placeholder="Enter your keywords here...\nOne keyword per line..."
    )
    
    # Parameters in a single row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_length = st.slider("Max text length in visualization", 10, 50, 20)
    
    with col2:
        n_clusters = st.slider("Number of clusters", 2, 10, 5)
    
    with col3:
        min_cluster_size = st.slider("Minimum cluster size", 1, 10, 3)
    
    # Process input
    if text_input:
        texts = [line.strip() for line in text_input.split('\n') if line.strip()]
        if texts:
            create_visualization(texts, max_length, n_clusters, min_cluster_size)
        else:
            st.info("Please enter some keywords to begin.")
    else:
        st.info("Please enter some keywords to begin.")

if __name__ == "__main__":
    if not Config.OPENAI_API_KEY or not Config.OPENROUTER_API_KEY:
        st.error("Please set your OPENAI_API_KEY and OPENROUTER_API_KEY in the .env file")
    else:
        main()
