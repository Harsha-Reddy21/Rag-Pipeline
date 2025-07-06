# ⚖️ Indian Legal Document Search System

A comprehensive comparison system for legal document retrieval using 4 different similarity methods. This system helps identify the most effective approach for finding relevant legal documents based on user queries.

## 🎯 Project Overview

This project implements a **RAG (Retrieval-Augmented Generation) pipeline** specifically designed for Indian legal documents. It compares four different similarity methods to determine which approach works best for legal document search and retrieval.

## 🔍 Similarity Methods Compared

### 1. **Cosine Similarity**
- Standard semantic matching using vector embeddings
- Measures the cosine of the angle between query and document vectors
- Best for: General semantic similarity queries

### 2. **Euclidean Distance**
- Geometric distance measurement in embedding space
- Uses negative distance for ranking (closer = better)
- Best for: Capturing different similarity patterns than cosine

### 3. **MMR (Maximal Marginal Relevance)**
- Balances relevance with diversity to reduce redundancy
- Configurable λ parameter (0.7 default) controls relevance vs diversity
- Best for: Avoiding redundant results, promoting variety

### 4. **Hybrid Similarity**
- Combines semantic similarity (60%) with legal entity matching (40%)
- Uses NLP entity extraction for legal terms
- Best for: Domain-specific legal document matching

## 🏛️ Sample Dataset

The system includes comprehensive Indian legal documents covering:

- **Income Tax Act** (Sections 80C, 80D)
- **GST Act** (Textile products, Professional services)
- **Property Law** (Registration, Stamp duty)
- **Court Procedures** (Fee structure, Appeal procedures)

## 📊 Evaluation Metrics

- **Precision@K**: Percentage of relevant documents in top K results
- **Diversity Score**: Variety of law types in results
- **Response Time**: Query processing speed
- **Top Score**: Highest similarity score achieved

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Legal-Document
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 💻 Usage

### Web Interface

1. **Open your browser** to `http://localhost:8501`
2. **Enter a legal query** or select from predefined test queries
3. **Upload documents** (PDF/Word) via the sidebar (optional)
4. **Adjust parameters** using the sidebar controls
5. **View results** in the 4-column comparison layout

### Test Queries

The system includes predefined test queries:
- "Income tax deduction for education"
- "GST rate for textile products"
- "Property registration process"
- "Court fee structure"

### Document Upload

- Supports **PDF** and **Word** documents
- Automatically extracts text and creates embeddings
- Integrates uploaded documents into the search corpus

## 🔧 Configuration

### Algorithm Parameters

- **MMR Lambda**: Controls relevance vs diversity balance (0.0-1.0)
- **Hybrid Weights**: Adjust cosine similarity vs entity matching weights

### Performance Tuning

- **Caching**: Models and embeddings are cached for performance
- **Batch Processing**: Efficient embedding generation
- **Responsive UI**: Real-time parameter adjustment

## 📈 Performance Analysis

The system provides comprehensive performance metrics:

### Precision@5 Comparison
- Measures accuracy of top 5 results
- Compares all 4 methods side-by-side
- Visual bar charts for easy comparison

### Diversity Score Analysis
- Measures variety of law types in results
- Important for comprehensive legal research
- Helps identify which method provides broader coverage

### Recommendations Engine
- Automatically identifies best-performing method
- Provides context-aware suggestions
- Highlights trade-offs between precision and diversity

## 🏗️ Architecture

```
📁 Legal-Document/
├── 📄 app.py              # Main Streamlit application
├── 📄 requirements.txt    # Python dependencies
├── 📄 README.md          # This file
├── 📄 experiments.ipynb  # Jupyter notebook experiments
└── 📁 myenv/             # Virtual environment
```

## 🧪 Experiments

The `experiments.ipynb` notebook contains:
- Initial algorithm development
- Performance testing
- Similarity method comparisons
- Proof-of-concept implementations

## 📋 Technical Stack

- **Frontend**: Streamlit
- **NLP**: sentence-transformers, spaCy
- **ML**: scikit-learn, numpy
- **Document Processing**: PyMuPDF, python-docx
- **Visualization**: Plotly, matplotlib
- **Vector Search**: FAISS (optional)

## 🔮 Future Enhancements

- [ ] **FAISS Integration**: Scale to larger document collections
- [ ] **Advanced NLP**: Legal-specific entity recognition
- [ ] **Multi-language Support**: Regional language documents
- [ ] **Query Expansion**: Automatic query enhancement
- [ ] **Relevance Feedback**: User rating integration
- [ ] **Export Features**: Save search results and analysis

## 📊 Results Summary

Based on comprehensive testing:

### Best for Precision
- **Cosine Similarity**: Excellent for semantic matching
- **Hybrid Similarity**: Best for domain-specific queries

### Best for Diversity
- **MMR**: Superior result variety
- **Euclidean Distance**: Alternative similarity patterns

### Recommendations
- **General Legal Search**: Use Hybrid Similarity
- **Exploratory Research**: Use MMR for diversity
- **Specific Document Matching**: Use Cosine Similarity
- **Alternative Perspectives**: Use Euclidean Distance
