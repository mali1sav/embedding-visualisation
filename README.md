# Embedding Visualisation Tool

A powerful tool for visualizing and analyzing text embeddings with SEO recommendations. This tool helps content editors understand and improve content alignment with search intent.

## Features

- **Interactive Visualization**: Display embeddings of intent, paragraphs, and keywords in a 2D space
- **Content Analysis**: Analyze semantic relationships between different text elements
- **SEO Recommendations**: Generate Thai-language SEO recommendations based on content analysis
- **Real-time Updates**: Visualizations and analysis persist between interactions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mali1sav/embedding-visualisation.git
cd embedding-visualisation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys (see `.env.example` for format):
```
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-...
```

## Usage

1. Run the core alignment tool:
```bash
streamlit run core_align.py
```

2. In the web interface:
   - Enter your search intent in Thai
   - Input your article content (paragraphs separated by double newlines)
   - Add target keywords (one per line)
   - Click "Generate Visualization" to see the embedding space
   - Click "Generate SEO Recommendations" for content improvement suggestions

## Visualization Guide

- **Gold Star**: Represents the main intent
- **Green Circles**: Article paragraphs
- **Blue Diamonds**: Target keywords
- Darker colors indicate higher similarity
- Closer positions indicate stronger semantic relationships

## Requirements

- Python 3.8+
- OpenAI API key
- OpenRouter API key
- macOS for system SSL certificates

## Notes

- The tool uses macOS system SSL certificates at `/etc/ssl/cert.pem`
- All SEO recommendations are generated in Thai language
- Visualization settings can be adjusted using the sliders

## License

MIT License
