# Smart Article Categorizer

A simple web app that compares 4 different embedding approaches for article classification.

## Features

- **4 Embedding Models**: Word2Vec, BERT, Sentence-BERT, OpenAI
- **Simple Interface**: Train models and classify articles
- **Real-time Results**: See predictions from all models instantly

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`

## Usage

1. **Train Models**: Load embedding models and train on your dataset
2. **Classify Articles**: Enter text and get predictions from all models

## Optional: OpenAI Integration

Set your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Dataset Format

CSV file with two columns:
- `text`: Article content
- `label`: Category (Tech, Finance, Healthcare, Sports, Politics, Entertainment)

That's it! Simple and effective. 