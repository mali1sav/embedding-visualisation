# Embedding Visualization Tool

A Streamlit-based tool for visualizing text embeddings using OpenAI's embedding model and t-SNE dimensionality reduction.

## Features

- Generate embeddings for text inputs using OpenAI's API
- Visualize embeddings in 2D/3D using t-SNE dimensionality reduction
- Interactive visualization with Plotly
- Nearest neighbor analysis for exploring similar texts
- Customizable visualization parameters

## Requirements

All required packages are listed in `requirements.txt`. Main dependencies include:
- OpenAI
- Streamlit
- Plotly
- NumPy
- scikit-learn
- pandas

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mali1sav/embedding-visualisation.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit app:
```bash
streamlit run ve.py
```

The app will open in your default web browser where you can:
1. Input your texts
2. Adjust visualization parameters
3. Toggle between 2D and 3D views
4. Explore nearest neighbors

## Note

Make sure you have a valid OpenAI API key and sufficient credits for generating embeddings.
