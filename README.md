# Keyword Embedding Visualization

A Streamlit application that visualizes keyword relationships using OpenAI embeddings and clustering. The app processes keywords, generates embeddings, clusters them, and creates an interactive visualization with cluster summaries.

## Features

- Text embedding using OpenAI's text-embedding-3-small model
- Hierarchical clustering with customizable parameters
- Interactive visualization using Plotly
- Cluster analysis with key terms and summaries
- Responsive web interface using Streamlit

## Requirements

- Python 3.8+
- OpenAI API key
- OpenRouter API key

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run ve.py
```

2. Enter keywords (one per line) in the text area
3. Adjust the visualization parameters if needed:
   - Max text length
   - Number of clusters
   - Minimum cluster size
4. View the interactive visualization and cluster analysis in the sidebar

## Note

This application uses OpenAI's API for embeddings and OpenRouter for LLM functionality. Make sure you have valid API keys and sufficient credits before running the application.
