# your_app.py

import os
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI  # Updated import
from sklearn.manifold import TSNE
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

@st.cache_data(show_spinner=False)
def get_embedding_cached(text, model="text-embedding-3-small"):
    """
    Get embedding for a single text using OpenAI API with caching.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding for text: {text}\n{e}")
        return None

def shorten_text(text, max_length=20):
    """
    Shorten text for labels while preserving meaning.
    """
    if len(text) <= max_length:
        return text
    words = text.split()
    if len(words) <= 2:
        return text[:max_length] + "..."
    return " ".join(words[:2]) + "..."

def process_text(text):
    """
    Convert text to readable format.
    """
    return text.strip().title()

def create_visualization(texts, max_length):
    if not texts:
        st.warning("No input texts provided.")
        return

    with st.spinner("Getting embeddings..."):
        embeddings = []
        processed_texts = []

        for text in texts:
            processed_text = process_text(text)
            processed_texts.append(processed_text)
            embedding = get_embedding_cached(processed_text)
            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            st.error("No embeddings were fetched. Please check your OpenAI API key and input texts.")
            return

    # Convert embeddings to numpy array
    embeddings_array = np.array(embeddings)

    with st.spinner("Reducing dimensions..."):
        n_samples = embeddings_array.shape[0]
        # Set perplexity to min(30, n_samples - 1) and at least 5
        perplexity = min(30, max(5, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)

    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings_2d)
    labels = clustering.labels_

    # Create DataFrame with texts and their cluster labels
    df = pd.DataFrame({
        'text': processed_texts,
        'cluster': labels,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'short_text': [shorten_text(text, max_length) for text in processed_texts]
    })

    # Display clustered keywords in the sidebar
    st.sidebar.markdown("## Keyword Groups")
    st.sidebar.markdown("*Click to copy group to clipboard*")

    unique_clusters = sorted(df['cluster'].unique())
    for cluster in unique_clusters:
        if cluster == -1:
            group_name = "Unclustered Keywords"
        else:
            group_name = f"Group {cluster + 1}"

        keywords = df[df['cluster'] == cluster]['text'].tolist()
        keywords_text = "\n".join(keywords)

        st.sidebar.text_area(
            group_name,
            keywords_text,
            height=100,
            key=f"cluster_{cluster}"
        )

    # Create DataFrame for plotting
    hover_text = [f"Text: {text}" for text in processed_texts]

    # Calculate similarity scores based on distances to nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=3).fit(embeddings_array)
    distances, _ = nbrs.kneighbors(embeddings_array)
    similarity_scores = 1 / (1 + np.mean(distances, axis=1))  # Convert distances to similarities

    # Create figure with custom layout
    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers+text',
        text=df['short_text'],
        hovertext=hover_text,
        hoverinfo='text',
        textposition='top center',
        marker=dict(
            size=10,
            color=similarity_scores,  # Color based on similarity to neighbors
            colorscale='Viridis',
            opacity=0.7,
            showscale=True,
            colorbar=dict(
                title='Similarity to neighbors',
                tickformat='.2f'
            )
        ),
        textfont=dict(size=10)
    ))

    # Update layout for better readability
    fig.update_layout(
        title={
            'text': 'Content Similarity Visualization',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title='t-SNE dimension 1',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'
        ),
        yaxis=dict(
            title='t-SNE dimension 2',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(211,211,211,0.3)'
        ),
        plot_bgcolor='white',
        width=None,
        height=700,
        showlegend=False,
        hovermode='closest'
    )

    # Add zoom and pan buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Reset View",
                         method="relayout",
                         args=[{"xaxis.autorange": True, "yaxis.autorange": True}]),
                ]
            )
        ]
    )

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

    # Add text search functionality
    search_term = st.text_input("Search for specific text:")
    if search_term:
        filtered_df = df[df['text'].str.contains(search_term, case=False)]
        if not filtered_df.empty:
            st.write("Found matches:")
            st.dataframe(filtered_df[['text']])
        else:
            st.write("No matches found.")

def main():
    st.set_page_config(page_title="Embedding Visualization", layout="wide")

    st.title("Text Embedding Visualization")

    # Add description
    st.markdown("""
    This tool creates a visualization of text similarities using OpenAI embeddings. 
    Similar texts will appear closer together and have similar colors in the visualization.
    
    ### Features:
    - Hover over points to see full text
    - Zoom and pan to explore clusters
    - Search for specific text below the visualization
    - Points are color-coded by similarity
    
    ### Instructions:
    1. Enter your texts below (one per line)
    2. Click 'Generate Visualization' to create the plot
    """)

    # Create text input area
    text_input = st.text_area(
        "Enter your texts (one per line):",
        height=200,
        placeholder="Enter text here...\nOne item per line..."
    )

    # Add file uploader as an alternative input method
    uploaded_file = st.file_uploader("Or upload a text file:", type=['txt'])

    if uploaded_file is not None:
        try:
            text_input = uploaded_file.getvalue().decode()
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

    # Add visualization options
    max_length = st.slider("Maximum label length:", 10, 50, 20)

    # Process input when button is clicked
    if st.button("Generate Visualization"):
        if text_input:
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            if len(texts) < 2:
                st.error("Please enter at least 2 texts to compare.")
            else:
                create_visualization(texts, max_length)
        else:
            st.warning("Please enter some text or upload a file.")

if __name__ == "__main__":
    if not os.getenv('OPENAI_API_KEY'):
        st.error("Please set your OPENAI_API_KEY in the .env file")
    else:
        main()
