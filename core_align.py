# core_alignment_app.py

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import httpx
import plotly.graph_objects as go
import ssl

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Check if API keys are present
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in the .env file.")
    st.stop()
if not OPENROUTER_API_KEY:
    st.error("Please set your OPENROUTER_API_KEY in the .env file.")
    st.stop()

# Create OpenAI client for embeddings with system SSL certificates
client_openai = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(
        verify="/etc/ssl/cert.pem"  # Default location for macOS system certificates
    )
)

# Create OpenAI client for chat completions via OpenRouter
client_openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://github.com/cascade",
        "X-Title": "Cascade"
    },
    http_client=httpx.Client(
        verify="/etc/ssl/cert.pem"  # Default location for macOS system certificates
    )
)

@st.cache_data(show_spinner=False)
def get_embedding_cached(text, model="text-embedding-3-large"):
    """
    Get embedding for a single text using OpenAI API.
    Using a well-known embedding model "text-embedding-3-large".
    """
    try:
        response = client_openai.embeddings.create(
            model=model,
            input=text
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

def interpolate_color(val, min_val, max_val, start_color, end_color):
    ratio = (val - min_val) / (max_val - min_val + 1e-10)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)
    r = int(start_rgb[0] + ratio*(end_rgb[0]-start_rgb[0]))
    g = int(start_rgb[1] + ratio*(end_rgb[1]-start_rgb[1]))
    b = int(start_rgb[2] + ratio*(end_rgb[2]-start_rgb[2]))
    return f"rgb({r},{g},{b})"

def call_openrouter(prompt: str, model: str = "openai/gpt-4o-2024-11-20", temperature=0.7, max_tokens=1000):
    """
    Calls OpenRouter API with the given prompt.
    System prompt in English, output in Thai.
    """
    system_prompt = (
        "You are an SEO expert. You will receive Thai content and an intent. Even though this system prompt is in English, produce all final answers in Thai."
        "Your job: analyze the provided content and main intent, then offer SEO recommendations."
        "These recommendations should include how to integrate relevant keywords into existing strong content, merge or rewrite paragraphs to improve topical coverage, remove irrelevant parts, and refine the flow for better alignment with the intent and user needs."
    )
    try:
        response = client_openrouter.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers={
                "HTTP-Referer": "https://github.com/cascade",
                "X-Title": "Cascade"
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error calling OpenRouter: {e}")
        return ""

def create_visualization(intent_texts, paragraphs, keywords, max_length, threshold, plot_key="viz_plot"):
    """
    Create a visualization of text clusters and show content relevance.
    Args:
        plot_key: Unique key for the plotly chart to avoid duplicate IDs
    """
    all_texts = intent_texts + paragraphs + keywords
    
    # Get embeddings
    embeddings = []
    for text in all_texts:
        embedding = get_embedding_cached(text)
        if embedding:
            embeddings.append(embedding)
        else:
            st.error(f"Could not get embedding for text: {text[:100]}...")
            return None

    if not embeddings:
        st.error("No embeddings generated. Please check your inputs.")
        return None

    embeddings_array = np.array(embeddings)

    # Use first intent embedding as reference, if available
    if intent_texts:
        intent_embedding = embeddings[0]
    else:
        intent_embedding = None

    similarities = []
    if intent_embedding is not None:
        for emb in embeddings:
            sim = cosine_sim(intent_embedding, emb)
            similarities.append(sim)
    else:
        similarities = [1.0] * len(embeddings)

    # Dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(embeddings)-1)))
    reduced_embeddings = tsne.fit_transform(embeddings_array)
    x_coords = reduced_embeddings[:, 0]
    y_coords = reduced_embeddings[:, 1]

    fig = go.Figure()

    # Plot Intent
    if intent_texts:
        intent_x = x_coords[:len(intent_texts)]
        intent_y = y_coords[:len(intent_texts)]
        intent_labels = [shorten_text(t, max_length) for t in intent_texts]
        fig.add_trace(go.Scatter(
            x=intent_x,
            y=intent_y,
            mode='markers+text',
            marker=dict(symbol='star', size=20, color='gold'),
            text=intent_labels,
            name='Intent',
            textposition="top center"
        ))

    # Plot Paragraphs
    start_idx = len(intent_texts)
    end_idx = start_idx + len(paragraphs)
    if paragraphs:
        para_x = x_coords[start_idx:end_idx]
        para_y = y_coords[start_idx:end_idx]
        para_labels = [shorten_text(p, max_length) for p in paragraphs]
        para_sims = similarities[start_idx:end_idx]
        para_colors = [interpolate_color(sim, 0, 1, '#90EE90', '#006400') for sim in para_sims]

        fig.add_trace(go.Scatter(
            x=para_x,
            y=para_y,
            mode='markers+text',
            marker=dict(size=15, color=para_colors),
            text=para_labels,
            name='Paragraphs',
            textposition="top center"
        ))

    # Plot Keywords
    start_kw = len(intent_texts) + len(paragraphs)
    if keywords:
        kw_x = x_coords[start_kw:]
        kw_y = y_coords[start_kw:]
        kw_labels = [shorten_text(k, max_length) for k in keywords]
        kw_sims = similarities[start_kw:]
        kw_colors = [interpolate_color(sim, 0, 1, '#ADD8E6', '#00008B') for sim in kw_sims]

        fig.add_trace(go.Scatter(
            x=kw_x,
            y=kw_y,
            mode='markers+text',
            marker=dict(symbol='diamond', size=12, color=kw_colors),
            text=kw_labels,
            name='Keywords',
            textposition="top center"
        ))

    fig.update_layout(
        title={
            'text': "Content Alignment Visualization",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        showlegend=True,
        width=1400,
        height=900,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=100, b=50),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    st.plotly_chart(fig, use_container_width=True, key=plot_key)

    # Show content relevance based on threshold
    if intent_texts:
        st.markdown("## Content Relevance Based on Similarity Threshold")

        # Paragraphs relevance
        if paragraphs:
            st.markdown("### Paragraphs ที่สอดคล้องกับ Intent")
            above_threshold_para = [p for p, s in zip(paragraphs, similarities[start_idx:end_idx]) if s >= threshold]
            below_threshold_para = [p for p, s in zip(paragraphs, similarities[start_idx:end_idx]) if s < threshold]

            st.markdown("#### ส่วนที่สอดคล้องเหนือ Threshold คือถือว่าดี")
            if above_threshold_para:
                for p in above_threshold_para:
                    st.write(f"- {p}")
            else:
                st.write("None")

            st.markdown("#### ส่วนที่มีความสอดคล้องต่ำกว่า Threshold ถือว่าควรปรับปรุง")
            if below_threshold_para:
                for p in below_threshold_para:
                    st.write(f"- {p}")
            else:
                st.write("None")

        # Keywords relevance
        if keywords:
            st.markdown("### คีย์เวิร์ด")
            above_threshold_kw = [k for k, s in zip(keywords, similarities[start_kw:]) if s >= threshold]
            below_threshold_kw = [k for k, s in zip(keywords, similarities[start_kw:]) if s < threshold]

            st.markdown("#### คีย์เวิร์ดที่สองคล้อง Main Intent และอยู่เหนือ Threshold ควรเก็บไว้")
            if above_threshold_kw:
                for k in above_threshold_kw:
                    st.write(f"- {k}")
            else:
                st.write("None")

            st.markdown("#### คีย์เวิร์ดที่ไม่ค่อยสอดคล้อง Main Intent อยู่ต่ำกว่า Threshold ควรพิจารณาเอาออก")
            if below_threshold_kw:
                for k in below_threshold_kw:
                    st.write(f"- {k}")
            else:
                st.write("None")

    return {
        'intent_texts': intent_texts,
        'paragraphs': paragraphs,
        'keywords': keywords,
        'similarities': similarities,
        'threshold': threshold
    }

def main():
    st.set_page_config(page_title="Core Alignment - Intent, Paragraphs, Keywords", layout="wide")

    # Initialize session state for all variables we need to persist
    if 'intent_input' not in st.session_state:
        st.session_state.intent_input = ""
    if 'paragraphs_input' not in st.session_state:
        st.session_state.paragraphs_input = ""
    if 'keywords_input' not in st.session_state:
        st.session_state.keywords_input = ""
    if 'visualization_done' not in st.session_state:
        st.session_state.visualization_done = False
    if 'viz_data' not in st.session_state:
        st.session_state.viz_data = None
    if 'current_visualization' not in st.session_state:
        st.session_state.current_visualization = None
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None
    if 'max_length' not in st.session_state:
        st.session_state.max_length = 20
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.6
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = False

    st.title("Core Alignment - Intent, Paragraphs, Keywords")

    st.markdown("""
    - **Star (Gold)**: Intent
    - **Green Circles**: Paragraphs
    - **Blue Diamonds**: Keywords
    - **Color Intensity**: Higher similarity = darker color
    - **Position**: Closer = more semantically similar
    """)

    # Input fields with session state
    intent_input = st.text_area(
        "Enter your search intent (Thai):",
        value=st.session_state.intent_input,
        height=100,
        placeholder="จุดประสงค์หลักของเนื้อหาคืออะไร?",
        key="intent_area"
    )

    paragraphs_input = st.text_area(
        "Enter your full article content (Thai, paragraphs separated by double newlines):",
        value=st.session_state.paragraphs_input,
        height=300,
        placeholder="วางบทความของคุณที่นี่...",
        key="paragraphs_area"
    )

    keywords_input = st.text_area(
        "Enter your target keywords (Thai, one per line):",
        value=st.session_state.keywords_input,
        height=100,
        placeholder="คีย์เวิร์ด1\nคีย์เวิร์ด2\n...",
        key="keywords_area"
    )

    max_length = st.slider("Maximum label length on chart:", 10, 50, st.session_state.max_length)
    threshold = st.slider("Similarity Threshold:", 0.0, 1.0, st.session_state.threshold, 0.01)

    # Generate Visualization button
    if st.button("Generate Visualization", key="viz_button"):
        st.session_state.intent_input = intent_input
        st.session_state.paragraphs_input = paragraphs_input
        st.session_state.keywords_input = keywords_input
        st.session_state.max_length = max_length
        st.session_state.threshold = threshold
        
        intent_texts = [i.strip() for i in intent_input.split('\n') if i.strip()]
        raw_paragraphs = [p.strip() for p in paragraphs_input.split('\n\n') if p.strip()]
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]

        if not intent_texts and not raw_paragraphs and not keywords:
            st.error("โปรดใส่ Intent, Paragraphs หรือ Keywords อย่างน้อยหนึ่งอย่าง")
        else:
            viz_data = create_visualization(intent_texts, raw_paragraphs, keywords, max_length, threshold, plot_key="initial_viz")
            if viz_data:
                st.session_state.visualization_done = True
                st.session_state.viz_data = viz_data
                st.session_state.current_visualization = viz_data

    # Always show visualization if it exists
    if st.session_state.current_visualization:
        intent_texts = [i.strip() for i in st.session_state.intent_input.split('\n') if i.strip()]
        raw_paragraphs = [p.strip() for p in st.session_state.paragraphs_input.split('\n\n') if p.strip()]
        keywords = [k.strip() for k in st.session_state.keywords_input.split('\n') if k.strip()]
        create_visualization(intent_texts, raw_paragraphs, keywords, st.session_state.max_length, st.session_state.threshold, plot_key="persistent_viz")

        # Show SEO recommendations button below visualization
        st.markdown("---")
        if st.button("Generate SEO Recommendations", key="seo_button"):
            st.session_state.show_recommendations = True
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
                    st.session_state.current_analysis = recommendations
                else:
                    st.error("ไม่สามารถสร้างคำแนะนำได้ โปรดตรวจสอบการตั้งค่าหรือ API key อีกครั้ง")

        # Always show current analysis if it exists
        if st.session_state.current_analysis and st.session_state.show_recommendations:
            st.markdown("## SEO Recommendations")
            st.write(st.session_state.current_analysis)

if __name__ == "__main__":
    main()
