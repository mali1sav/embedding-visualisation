# core_alignment_app.py

import os
import numpy as np
import streamlit as st
from openai import OpenAI
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import httpx
from dotenv import load_dotenv
import re
from sklearn.cluster import DBSCAN
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with appropriate SSL verification
ssl_verify = "/etc/ssl/cert.pem" if os.path.exists("/etc/ssl/cert.pem") else True

# Initialize HTTP client
http_client = httpx.Client(
    base_url="https://openrouter.ai/api/v1",
    verify=ssl_verify,
    timeout=60.0,
    headers={
        "HTTP-Referer": "https://embedding-visualisation.streamlit.app",  # Update with your Streamlit app URL
        "X-Title": "Embedding Visualisation",
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"
    }
)

client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    http_client=http_client,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://embedding-visualisation.streamlit.app",  # Update with your Streamlit app URL
        "X-Title": "Embedding Visualisation"
    }
)

@st.cache_data(show_spinner=False)
def get_embedding_cached(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error fetching embedding for text:\n{e}")
        return None

def process_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def shorten_text(text, max_length=20):
    if len(text) <= max_length:
        return text
    words = text.split()
    if len(words) <= 2:
        return text[:max_length] + "..."
    return " ".join(words[:2]) + "..."

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def create_visualisation(intent_texts, paragraphs, keywords, max_length, threshold):
    # Combine all texts and get embeddings
    all_texts = []
    categories = []
    embeddings = []

    for it in intent_texts:
        p_it = process_text(it)
        emb = get_embedding_cached(p_it)
        if emb is not None:
            all_texts.append(p_it)
            categories.append("Intent")
            embeddings.append(emb)

    paragraph_embeddings = []
    for p in paragraphs:
        p_p = process_text(p)
        emb = get_embedding_cached(p_p)
        if emb is not None:
            all_texts.append(p_p)
            categories.append("Paragraph")
            embeddings.append(emb)
            paragraph_embeddings.append(emb)

    for kw in keywords:
        p_kw = process_text(kw)
        emb = get_embedding_cached(p_kw)
        if emb is not None:
            all_texts.append(p_kw)
            categories.append("Keyword")
            embeddings.append(emb)

    if len(embeddings) < 2:
        st.error("Not enough embeddings to visualise. Please provide more data.")
        return

    embeddings_array = np.array(embeddings)

    # Compute core embedding from paragraphs (or all if none exist)
    if len(paragraph_embeddings) > 0:
        core_embedding = np.mean(np.array(paragraph_embeddings), axis=0)
    else:
        core_embedding = np.mean(embeddings_array, axis=0)

    similarities_to_core = [cosine_sim(e, core_embedding) for e in embeddings_array]

    # Dimensionality reduction
    with st.spinner("Reducing dimensions with t-SNE..."):
        n_samples = embeddings_array.shape[0]
        perplexity = min(30, max(5, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_array)

    # Clustering with DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings_2d)
    labels = clustering.labels_

    df = pd.DataFrame({
        'text': all_texts,
        'category': categories,
        'cluster': labels,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'short_text': [shorten_text(t, max_length) for t in all_texts],
        'similarity_to_core': similarities_to_core
    })

    # Assign colors based on category and similarity
    def interpolate_color(val, min_val, max_val, start_color, end_color):
        # val in [min_val, max_val], map to [0,1]
        ratio = (val - min_val) / (max_val - min_val + 1e-10)
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)

        # Interpolate
        r = int(start_rgb[0] + ratio*(end_rgb[0]-start_rgb[0]))
        g = int(start_rgb[1] + ratio*(end_rgb[1]-start_rgb[1]))
        b = int(start_rgb[2] + ratio*(end_rgb[2]-start_rgb[2]))
        return f"rgb({r},{g},{b})"

    # Define color ranges for each category
    intent_start, intent_end = "#FFFACD", "#FFD700"    # Light LemonChiffon to Gold (Yellow)
    para_start, para_end = "#E0FFE0", "#006400"        # Light Green to Dark Green
    kw_start, kw_end     = "#E0F0FF", "#00008B"        # Light Blue to Dark Blue

    sim_min = df['similarity_to_core'].min()
    sim_max = df['similarity_to_core'].max()

    def get_color(cat, sim):
        if cat == "Intent":
            return interpolate_color(sim, sim_min, sim_max, intent_start, intent_end)
        elif cat == "Paragraph":
            return interpolate_color(sim, sim_min, sim_max, para_start, para_end)
        elif cat == "Keyword":
            return interpolate_color(sim, sim_min, sim_max, kw_start, kw_end)
        else:
            return "#cccccc"

    df['color'] = [get_color(cat, sim) for cat, sim in zip(df['category'], df['similarity_to_core'])]

    # Create hover text
    def make_hover_text(row):
        return f"Category: {row['category']}\nText: {row['text']}\nSimilarity to core: {row['similarity_to_core']:.2f}"

    fig = go.Figure()

    # Plot Intent (star, yellow)
    df_intent = df[df['category'] == 'Intent']
    if not df_intent.empty:
        fig.add_trace(go.Scatter(
            x=df_intent['x'],
            y=df_intent['y'],
            mode='markers+text',
            text=df_intent['short_text'],
            hovertext=[make_hover_text(r) for _, r in df_intent.iterrows()],
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                size=20,  # Bigger star
                color=df_intent['color'],
                symbol='star',
                opacity=0.9
            ),
            textfont=dict(size=14, family='Arial Black')
        ))

    # Plot Paragraphs (circle, green)
    df_paragraphs = df[df['category'] == 'Paragraph']
    if not df_paragraphs.empty:
        fig.add_trace(go.Scatter(
            x=df_paragraphs['x'],
            y=df_paragraphs['y'],
            mode='markers+text',
            text=df_paragraphs['short_text'],
            hovertext=[make_hover_text(r) for _, r in df_paragraphs.iterrows()],
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                size=12,
                color=df_paragraphs['color'],
                symbol='circle',
                opacity=0.8
            ),
            textfont=dict(size=12)
        ))

    # Plot Keywords (diamond, blue)
    df_keywords = df[df['category'] == 'Keyword']
    if not df_keywords.empty:
        fig.add_trace(go.Scatter(
            x=df_keywords['x'],
            y=df_keywords['y'],
            mode='markers+text',
            text=df_keywords['short_text'],
            hovertext=[make_hover_text(r) for _, r in df_keywords.iterrows()],
            hoverinfo='text',
            textposition='top center',
            marker=dict(
                size=12,
                color=df_keywords['color'],
                symbol='diamond',
                opacity=0.8
            ),
            textfont=dict(size=12)
        ))

    fig.update_layout(
        title={
            'text': 'Core Alignment - Intent, Paragraphs, Keywords',
            'y':0.95,
            'x':0.5,
            'xanchor':'center',
            'yanchor':'top',
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
        height=750,
        showlegend=False,
        hovermode='closest'
    )

    # Add reset view button
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

    st.plotly_chart(fig, use_container_width=True)

    # Show lists of paragraphs/keywords above and below threshold
    # Filter based on threshold
    above_threshold_para = df[(df['category'] == 'Paragraph') & (df['similarity_to_core'] > threshold)]
    above_threshold_kw = df[(df['category'] == 'Keyword') & (df['similarity_to_core'] > threshold)]

    below_threshold_para = df[(df['category'] == 'Paragraph') & (df['similarity_to_core'] <= threshold)]
    below_threshold_kw = df[(df['category'] == 'Keyword') & (df['similarity_to_core'] <= threshold)]

    # Make the header green
    st.markdown(f"<h3 style='color: green;'>Content Relevance Based on Similarity > {threshold}</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Paragraphs Above Threshold**")
        if not above_threshold_para.empty:
            for t in above_threshold_para['text']:
                st.write(t)
        else:
            st.write("None")

    with col2:
        st.markdown("**Keywords Above Threshold**")
        if not above_threshold_kw.empty:
            for t in above_threshold_kw['text']:
                st.write(t)
        else:
            st.write("None")

    st.markdown("---")

    # Make the header red for below threshold
    st.markdown(f"<h3 style='color: red;'>Content Relevance Based on Similarity ≤ {threshold}</h3>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Paragraphs Below or Equal to Threshold**")
        if not below_threshold_para.empty:
            for t in below_threshold_para['text']:
                st.write(t)
        else:
            st.write("None")

    with col4:
        st.markdown("**Keywords Below or Equal to Threshold**")
        if not below_threshold_kw.empty:
            for t in below_threshold_kw['text']:
                st.write(t)
        else:
            st.write("None")

    return {
        'intent_texts': intent_texts,
        'paragraphs': paragraphs,
        'keywords': keywords
    }

def call_openrouter(prompt):
    try:
        response = client.chat.completions.create(
            model="openai/o1",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            headers={
                "HTTP-Referer": "https://embedding-visualisation.streamlit.app",  # Update with your Streamlit app URL
                "X-Title": "Embedding Visualisation"
            }
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error fetching recommendations:\n{e}")
        return None

def main():
    st.set_page_config(page_title="Core Alignment - Intent, Paragraphs, Keywords", layout="wide")

    st.title("Core Alignment - Intent, Paragraphs, Keywords")

    st.markdown("""    
    - **Color Intensity**: Represents similarity to the core topic. Darker shades indicate higher similarity.
    - **Positions**: Reflect semantic similarity; closer points share more meaning.
    - **Content Relevance Based on Similarity Threshold**: Lists of paragraphs and keywords above and below the chosen similarity threshold to help you decide which content to retain or refine for better topical relevance.
    """)

    intent_input = st.text_area(
        "Enter your search intent (Thai):",
        height=100,
        placeholder="จุดประสงค์หลักของเนื้อหาคืออะไร?"
    )

    paragraphs_input = st.text_area(
        "Enter your full article content (Thai, paragraphs separated by double newlines):",
        height=300,
        placeholder="วางบทความของคุณที่นี่..."
    )

    keywords_input = st.text_area(
        "Enter your target keywords (Thai, one per line):",
        height=100,
        placeholder="คีย์เวิร์ด1\nคีย์เวิร์ด2\n..."
    )

    max_length = st.slider("Maximum label length on chart:", 10, 50, 20)
    threshold = st.slider("Similarity Threshold:", 0.0, 1.0, 0.7, 0.01)

    if st.button("Generate visualisation"):
        intent_texts = [i.strip() for i in intent_input.split('\n') if i.strip()]
        raw_paragraphs = [p.strip() for p in paragraphs_input.split('\n\n') if p.strip()]
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

        if not intent_texts and not raw_paragraphs and not keywords:
            st.error("Please provide at least some intent, paragraphs, or keywords.")
        else:
            viz_data = create_visualisation(intent_texts, raw_paragraphs, keywords, max_length, threshold)
            if viz_data:
                st.session_state.visualization_done = True
                st.session_state.viz_data = viz_data

        # Show SEO recommendations button below visualization
        if st.session_state.get('visualization_done', False):
            st.markdown("---")
            if st.button("Generate SEO Recommendations"):
                data = st.session_state.viz_data
                prompt = f"""
                Content Analysis Request:
                
                1. Main Intent:
                {' '.join(data['intent_texts'])}
                
                2. Current Content:
                {' '.join(data['paragraphs'])}
                
                3. Target Keywords:
                {', '.join(data['keywords'])}
                
                Please analyze the current content and suggest how to:

                1. Blend and integrate relevant target keywords into already strong sections of the existing content.
                2. Identify paragraphs that can be merged or restructured to improve flow and clarity, rather than just adding new content.
                3. Suggest which paragraphs could be removed if they do not serve the main intent or user needs.
                4. Provide guidance on rewriting certain parts to more naturally incorporate keywords and related concepts while maintaining a coherent narrative.
                
                Note: All recommendations should be practical, specific, and suitable for a Thai content editor looking to refine the existing piece for better search visibility and user satisfaction.
                """
                
                with st.spinner("กำลังสร้างคำแนะนำ SEO..."):
                    recommendations = call_openrouter(prompt)
                    if recommendations:
                        st.markdown("## SEO Recommendations")
                        st.write(recommendations)
                    else:
                        st.error("ไม่สามารถสร้างคำแนะนำได้ โปรดตรวจสอบการตั้งค่าหรือ API key อีกครั้ง")

if __name__ == "__main__":
    if not os.getenv('OPENROUTER_API_KEY'):
        st.error("OPENROUTER_API_KEY not found in environment variables. Please add it to your .env file or Streamlit secrets.")
        st.stop()
    main()
