"""
LangChain Integration Example for Intelligent Document Chunking
"""

import os
from typing import List, Dict, Any, Tuple
import re

# Import LangChain components
from langchain.text_splitter import TextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document

# Import our document processing functions
# Note: In a real implementation, these would be imported from app.py
# Here we're redefining them for demonstration purposes

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

def process_document(document_text: str, metadata: Dict[str, Any] = None) -> Tuple[List[str], str]:
    """
    Process a document through the intelligent chunking pipeline.
    """
    # Detect document type
    doc_type = detect_document_type(document_text, metadata)
    
    # Apply chunking
    chunks = adaptive_chunk_document(document_text, metadata)
    
    return chunks, doc_type

# Create a custom LangChain text splitter using our adaptive chunking
class AdaptiveDocumentSplitter(TextSplitter):
    """
    Custom LangChain text splitter that uses our adaptive chunking logic.
    """
    def __init__(self):
        super().__init__(chunk_size=1000, chunk_overlap=0)
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text using our adaptive chunking strategy.
        """
        chunks, _ = process_document(text)
        return chunks
    
    def create_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[Document]:
        """
        Create LangChain documents from texts and optional metadatas.
        """
        if not metadatas:
            metadatas = [{} for _ in texts]
            
        documents = []
        
        for i, text in enumerate(texts):
            metadata = metadatas[i] if i < len(metadatas) else {}
            chunks, doc_type = process_document(text, metadata)
            
            # Add document type to metadata
            for chunk in chunks:
                chunk_metadata = metadata.copy()
                chunk_metadata["doc_type"] = doc_type
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
        return documents

# Example usage
def main():
    print("LangChain Integration Example")
    print("============================")
    
    # Initialize the embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize our custom text splitter
    splitter = AdaptiveDocumentSplitter()
    
    # Example documents
    example_documents = {
        "technical.md": """
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
        "policy.md": """
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
"""
    }
    
    # Create directory for example files if it doesn't exist
    os.makedirs("example_docs", exist_ok=True)
    
    # Write example documents to files
    for filename, content in example_documents.items():
        with open(f"example_docs/{filename}", "w") as f:
            f.write(content)
    
    print(f"Created example documents in 'example_docs' directory")
    
    # Load documents using LangChain's loaders
    loader = DirectoryLoader("example_docs", glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents")
    
    # Process documents with our custom splitter
    split_docs = splitter.split_documents(documents)
    
    print(f"Split into {len(split_docs)} chunks")
    print("\nChunk examples:")
    for i, doc in enumerate(split_docs[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1} (Type: {doc.metadata.get('doc_type', 'unknown')}):")
        print(f"{doc.page_content[:100]}...")
    
    # Create vector store
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    
    # Example queries
    example_queries = [
        "How do I authenticate with the API?",
        "What are the data protection requirements?",
        "What should I do in case of a security incident?"
    ]
    
    print("\nExample Queries:")
    for query in example_queries:
        print(f"\nQuery: {query}")
        results = vectorstore.similarity_search(query, k=1)
        
        if results:
            doc = results[0]
            print(f"Result (Type: {doc.metadata.get('doc_type', 'unknown')}):")
            print(f"{doc.page_content[:200]}...")
        else:
            print("No results found")
    
    print("\nLangChain integration complete!")

if __name__ == "__main__":
    main() 