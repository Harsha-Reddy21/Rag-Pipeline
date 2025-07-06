import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import io
import base64
from typing import List, Dict, Any, Tuple
import json
import time

# Set page configuration
st.set_page_config(
    page_title="Intelligent Document Chunking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stProgress .st-bo {
        background-color: #3B82F6;
    }
    .document-chunk {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    .code-chunk {
        background-color: #ECFDF5;
        border-left: 4px solid #059669;
    }
    .policy-chunk {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
    }
    .tutorial-chunk {
        background-color: #F3E8FF;
        border-left: 4px solid #7C3AED;
    }
</style>
""", unsafe_allow_html=True)

# Document Classification Functions
def detect_document_type(document_text: str, metadata: Dict[str, Any] = None) -> str:
    """
    Detect the type of document based on content patterns and metadata.
    """
    # Simple pattern matching for document classification
    if metadata and "source" in metadata:
        if "github" in metadata["source"].lower():
            return "code"
        if "confluence" in metadata["source"].lower():
            return "technical"
        if "jira" in metadata["source"].lower():
            return "support"
            
    # Content-based classification
    code_pattern = re.compile(r'```[\w]*\n|def\s+\w+\(|class\s+\w+[:\(]|import\s+\w+|from\s+\w+\s+import')
    if code_pattern.search(document_text):
        return "code"
    
    if re.search(r'step\s+\d+|procedure|how\s+to|tutorial', document_text, re.IGNORECASE):
        return "tutorial"
        
    if re.search(r'policy|requirement|compliance|must\s+not|shall|should', document_text, re.IGNORECASE):
        return "policy"
        
    # Default to technical documentation
    return "technical"

# Chunking Strategy Functions
def chunk_technical_document(document_text: str) -> List[str]:
    """
    Chunk technical documents by sections and paragraphs.
    """
    # Split by headers (##, ###) and then by paragraphs
    sections = re.split(r'(?m)^#{2,3}\s+', document_text)
    chunks = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Further split large sections into paragraphs
        if len(section) > 1000:
            paragraphs = re.split(r'\n\n+', section)
            chunks.extend([p for p in paragraphs if p.strip()])
        else:
            chunks.append(section)
            
    return chunks

def chunk_code_document(document_text: str) -> List[str]:
    """
    Chunk code documents preserving function/class definitions.
    """
    # Identify code blocks
    code_blocks = re.split(r'```(?:\w+)?\n', document_text)
    chunks = []
    
    for i, block in enumerate(code_blocks):
        if i % 2 == 1:  # This is a code block
            # Keep code blocks intact
            chunks.append(block)
        else:  # This is regular text
            # Split regular text by paragraphs
            text_chunks = re.split(r'\n\n+', block)
            chunks.extend([chunk for chunk in text_chunks if chunk.strip()])
            
    return chunks

def chunk_policy_document(document_text: str) -> List[str]:
    """
    Chunk policy documents preserving policy items and requirements.
    """
    # Split by policy sections and numbered items
    sections = re.split(r'(?m)^#{2,3}\s+|^\d+\.\s+', document_text)
    chunks = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # Keep policy requirements together
        if len(section) > 1200:
            subsections = re.split(r'\n\n+', section)
            chunks.extend([s for s in subsections if s.strip()])
        else:
            chunks.append(section)
            
    return chunks

def chunk_tutorial_document(document_text: str) -> List[str]:
    """
    Chunk tutorial documents preserving step sequences.
    """
    # Split by step headers but keep sequential steps together
    step_pattern = re.compile(r'(?m)^Step\s+\d+:|^#{2,3}\s+Step\s+\d+')
    sections = step_pattern.split(document_text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        # Group related steps together
        if len(current_chunk) + len(section) < 1000:
            current_chunk += section
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

# Main Chunking Function
def adaptive_chunk_document(document_text: str, metadata: Dict[str, Any] = None) -> List[str]:
    """
    Apply adaptive chunking based on document type.
    """
    # Detect document type
    doc_type = detect_document_type(document_text, metadata)
    
    # Apply appropriate chunking strategy
    if doc_type == "code":
        return chunk_code_document(document_text)
    elif doc_type == "policy":
        return chunk_policy_document(document_text)
    elif doc_type == "tutorial":
        return chunk_tutorial_document(document_text)
    else:  # technical or default
        return chunk_technical_document(document_text)

# Processing Pipeline
def process_document(document_text: str, metadata: Dict[str, Any] = None) -> Tuple[List[str], str]:
    """
    Process a document through the intelligent chunking pipeline.
    """
    # Detect document type
    doc_type = detect_document_type(document_text, metadata)
    
    # Apply chunking
    chunks = adaptive_chunk_document(document_text, metadata)
    
    return chunks, doc_type

# Performance Evaluation
def evaluate_chunking_quality(chunks: List[str], original_text: str) -> Dict[str, float]:
    """
    Evaluate the quality of document chunking.
    """
    # Calculate basic metrics
    total_chunks = len(chunks)
    avg_chunk_size = sum(len(chunk) for chunk in chunks) / total_chunks if total_chunks > 0 else 0
    coverage = sum(len(chunk) for chunk in chunks) / len(original_text) if original_text else 0
    
    # Check for content preservation
    content_preserved = all(chunk in original_text for chunk in chunks)
    
    # Calculate chunk size distribution
    sizes = [len(chunk) for chunk in chunks]
    size_variation = max(sizes) - min(sizes) if sizes else 0
    
    return {
        "total_chunks": total_chunks,
        "avg_chunk_size": avg_chunk_size,
        "coverage": coverage,
        "content_preserved": content_preserved,
        "size_variation": size_variation,
        "min_size": min(sizes) if sizes else 0,
        "max_size": max(sizes) if sizes else 0
    }

# Simulated Vector Store for Demo
class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.doc_types = []
        
    def add_chunks(self, chunks: List[str], doc_type: str):
        self.chunks.extend(chunks)
        self.doc_types.extend([doc_type] * len(chunks))
        
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float, str]]:
        # Simple keyword search for demo purposes
        results = []
        query_terms = query.lower().split()
        
        for i, chunk in enumerate(self.chunks):
            score = 0
            for term in query_terms:
                if term.lower() in chunk.lower():
                    score += 1
            
            if score > 0:
                results.append((chunk, score / len(query_terms), self.doc_types[i]))
                
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

# Initialize vector store
vector_store = SimpleVectorStore()

# Example documents
example_documents = {
    "technical": """
# Technical Guide to API Integration

## Overview
This document explains how to integrate with our REST API.

## Authentication
Authentication uses OAuth 2.0. Here's an example:

```python
import requests

def get_token(client_id, client_secret):
    url = "https://api.example.com/oauth/token"
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, data=payload)
    return response.json()["access_token"]
```

## Endpoints
The following endpoints are available:
- GET /users
- POST /users
- GET /products

## Error Handling
Handle errors by checking the status code.
""",
    
    "code": """
# Code Repository Documentation

## Main Module

```python
class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        
    def process(self, document):
        # Process the document
        return processed_document
```

## Utility Functions

```python
def parse_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)
```
""",
    
    "policy": """
# Corporate Security Policy

## Data Protection Requirements

1. All customer data must be encrypted at rest and in transit.
2. Access to production data requires two-factor authentication.
3. Data retention periods must comply with local regulations.

## Incident Response

In case of a security incident, the following steps must be taken:
1. Isolate affected systems
2. Notify the security team within 1 hour
3. Document all actions taken
""",
    
    "tutorial": """
# How to Deploy the Application

## Step 1: Environment Setup
Install the required dependencies:
- Node.js v14+
- MongoDB
- Redis

## Step 2: Configuration
Copy the example config file and update with your settings:
```
cp config.example.json config.json
```

## Step 3: Database Initialization
Run the setup script:
```
npm run db:setup
```

## Step 4: Start the Application
Launch the application with:
```
npm start
```
"""
}

# Main Application UI
def main():
    st.markdown('<p class="main-header">Intelligent Document Chunking</p>', unsafe_allow_html=True)
    st.markdown("""
    Adaptive chunking system for enterprise knowledge management that automatically detects document types 
    and applies appropriate chunking strategies to improve knowledge retrieval.
    """)
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Document Processing", "Batch Processing", "Search Demo", "Performance Metrics"])
    
    with tab1:
        st.markdown('<p class="sub-header">Document Processing</p>', unsafe_allow_html=True)
        
        # Document input
        doc_source = st.selectbox(
            "Document Source",
            ["Custom Input", "Confluence", "GitHub Wiki", "Jira Ticket", "Example Documents"]
        )
        
        if doc_source == "Example Documents":
            example_choice = st.selectbox(
                "Select Example Document",
                list(example_documents.keys())
            )
            document_text = example_documents[example_choice]
            metadata = {"source": f"{example_choice}_source"}
        else:
            document_text = st.text_area("Document Content", height=300)
            metadata = {"source": doc_source}
        
        # Process button
        if st.button("Process Document"):
            if document_text:
                with st.spinner("Processing document..."):
                    # Process the document
                    chunks, doc_type = process_document(document_text, metadata)
                    metrics = evaluate_chunking_quality(chunks, document_text)
                    
                    # Add to vector store
                    vector_store.add_chunks(chunks, doc_type)
                    
                    # Display results
                    st.success(f"Document processed successfully! Detected type: **{doc_type.upper()}**")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", metrics["total_chunks"])
                    with col2:
                        st.metric("Avg Chunk Size", f"{metrics['avg_chunk_size']:.2f}")
                    with col3:
                        st.metric("Content Coverage", f"{metrics['coverage']*100:.2f}%")
                    
                    # Display chunks
                    st.markdown("### Document Chunks")
                    for i, chunk in enumerate(chunks):
                        chunk_class = f"document-chunk {doc_type}-chunk" if doc_type in ["code", "policy", "tutorial"] else "document-chunk"
                        st.markdown(f"""
                        <div class="{chunk_class}">
                            <strong>Chunk {i+1}</strong> ({len(chunk)} chars)<br>
                            <pre>{chunk[:200]}{"..." if len(chunk) > 200 else ""}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualize chunk sizes
                    st.markdown("### Chunk Size Distribution")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    chunk_sizes = [len(chunk) for chunk in chunks]
                    ax.bar(range(len(chunk_sizes)), chunk_sizes)
                    ax.set_xlabel("Chunk Index")
                    ax.set_ylabel("Size (characters)")
                    ax.set_title(f"Chunk Size Distribution for {doc_type.capitalize()} Document")
                    st.pyplot(fig)
            else:
                st.error("Please enter document content.")
    
    with tab2:
        st.markdown('<p class="sub-header">Batch Processing</p>', unsafe_allow_html=True)
        
        # Upload multiple files
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Process Batch"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                
                for i, file in enumerate(uploaded_files):
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
                    
                    # Read file content
                    content = file.read().decode("utf-8")
                    
                    # Process document
                    chunks, doc_type = process_document(content, {"source": file.name})
                    metrics = evaluate_chunking_quality(chunks, content)
                    
                    # Add to results
                    results.append({
                        "filename": file.name,
                        "doc_type": doc_type,
                        "chunks": len(chunks),
                        "avg_size": metrics["avg_chunk_size"]
                    })
                    
                    # Add to vector store
                    vector_store.add_chunks(chunks, doc_type)
                    
                    # Simulate processing time
                    time.sleep(0.5)
                
                # Display results table
                st.success(f"Processed {len(uploaded_files)} documents successfully!")
                st.dataframe(pd.DataFrame(results))
                
                # Download results button
                csv = pd.DataFrame(results).to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="batch_results.csv">Download Results CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<p class="sub-header">Search Demo</p>', unsafe_allow_html=True)
        
        # Search interface
        search_query = st.text_input("Search Query")
        
        if st.button("Search") and search_query:
            with st.spinner("Searching..."):
                # Perform search
                results = vector_store.search(search_query)
                
                # Display results
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    for i, (chunk, score, doc_type) in enumerate(results):
                        chunk_class = f"document-chunk {doc_type}-chunk" if doc_type in ["code", "policy", "tutorial"] else "document-chunk"
                        st.markdown(f"""
                        <div class="{chunk_class}">
                            <strong>Result {i+1}</strong> (Score: {score:.2f}, Type: {doc_type})<br>
                            <pre>{chunk[:300]}{"..." if len(chunk) > 300 else ""}</pre>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No results found. Try a different query.")
        
    with tab4:
        st.markdown('<p class="sub-header">Performance Metrics</p>', unsafe_allow_html=True)
        
        # Generate some mock performance data
        if len(vector_store.chunks) > 0:
            # Chunking metrics by document type
            doc_types = ["technical", "code", "policy", "tutorial"]
            chunk_counts = [sum(1 for dt in vector_store.doc_types if dt == doc_type) for doc_type in doc_types]
            
            # Display metrics
            st.markdown("### Document Type Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(doc_types, chunk_counts)
            ax.set_xlabel("Document Type")
            ax.set_ylabel("Number of Chunks")
            ax.set_title("Chunks by Document Type")
            st.pyplot(fig)
            
            # Mock retrieval accuracy metrics
            st.markdown("### Retrieval Accuracy Metrics")
            
            accuracy_data = {
                "Uniform Chunking": 0.65,
                "Adaptive Chunking": 0.87
            }
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(accuracy_data.keys(), accuracy_data.values())
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Accuracy Score")
            ax.set_title("Retrieval Accuracy Comparison")
            
            # Add percentage labels
            for i, v in enumerate(accuracy_data.values()):
                ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
                
            st.pyplot(fig)
            
            # Metrics cards
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>Total Documents</h4>
                    <h2>42</h2>
                    <p>+12% from last month</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>Total Chunks</h4>
                    <h2>187</h2>
                    <p>+24% from last month</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>Avg. Query Success</h4>
                    <h2>87%</h2>
                    <p>+18% from uniform chunking</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Process some documents to see performance metrics.")

if __name__ == "__main__":
    main() 