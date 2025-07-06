import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import fitz  # PyMuPDF
from docx import Document
import io
import time
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

# Page config
st.set_page_config(
    page_title="Indian Legal Document Search System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models (cached for performance)
@st.cache_resource
def load_models():
    """Load sentence transformer model"""
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return sentence_model

# Initialize models
sentence_model = load_models()

# Sample Indian Legal Documents Dataset
@st.cache_data
def get_sample_documents():
    """Return sample Indian legal documents for testing"""
    return [
        {
            "id": "doc1",
            "title": "Section 80C - Income Tax Deductions",
            "body": "Under Section 80C of the Income Tax Act, 1961, taxpayers can claim deductions up to Rs. 1,50,000 for investments in specified instruments including life insurance premiums, provident fund contributions, equity-linked savings schemes (ELSS), National Savings Certificate (NSC), Public Provident Fund (PPF), home loan principal repayment, children's tuition fees, and Sukanya Samriddhi Yojana. The deduction is available for both individuals and Hindu Undivided Families (HUFs). Educational expenses for children, housing loan principal payments, and life insurance premiums are popular choices under this section.",
            "law_type": "Income Tax Act",
            "section": "80C",
            "year": "1961"
        },
        {
            "id": "doc2", 
            "title": "GST Rates on Textile Products",
            "body": "Under the Goods and Services Tax (GST) regime, textile products attract different tax rates. Cotton yarn, fabrics, and readymade garments attract 5% GST as per Schedule I of the GST Act. However, branded textiles with sale value exceeding Rs. 1000 per piece attract 12% GST. Silk textiles, synthetic fabrics, and luxury textile items may attract 18% GST. Raw cotton and cotton waste are exempted from GST. The textile industry benefits from various GST exemptions and reduced rates to promote manufacturing and exports.",
            "law_type": "GST Act",
            "section": "Schedule I",
            "year": "2017"
        },
        {
            "id": "doc3",
            "title": "Property Registration Process in India",
            "body": "Property registration in India is mandatory under the Registration Act, 1908. The process involves payment of stamp duty (varies by state, typically 3-7% of property value), registration fees (typically 1% of property value), and completion of registration within 4 months of execution. Required documents include sale deed, property title documents, encumbrance certificate, property tax receipts, and identity proofs. The registration must be done at the sub-registrar office in the jurisdiction where the property is located. Online registration facilities are available in many states through e-registration portals.",
            "law_type": "Property Law",
            "section": "Registration Act",
            "year": "1908"
        },
        {
            "id": "doc4",
            "title": "Court Fee Structure for Civil Cases",
            "body": "Court fees for civil cases in India are governed by the Court Fees Act, 1870. The fee structure is based on the value of the suit: for suits up to Rs. 1 lakh, the fee is 7.5% of the suit value; for suits between Rs. 1-5 lakhs, it's Rs. 7,500 plus 5% of the amount exceeding Rs. 1 lakh; for suits above Rs. 5 lakhs, it's Rs. 27,500 plus 2% of the amount exceeding Rs. 5 lakhs. Additional fees apply for applications, appeals, and other proceedings. Some states have different fee structures as per their respective amendments to the Court Fees Act.",
            "law_type": "Court Fees Act",
            "section": "Schedule I",
            "year": "1870"
        },
        {
            "id": "doc5",
            "title": "Section 80D - Medical Insurance Deductions",
            "body": "Section 80D of the Income Tax Act allows deductions for medical insurance premiums paid for self, spouse, children, and parents. The deduction limit is Rs. 25,000 for individuals under 60 years and Rs. 50,000 for senior citizens. An additional Rs. 25,000 (Rs. 50,000 for senior citizen parents) can be claimed for parents' medical insurance. Preventive health check-up expenses up to Rs. 5,000 are also deductible. The premium must be paid through approved modes (not cash) and the insurance must be with an approved insurer.",
            "law_type": "Income Tax Act",
            "section": "80D",
            "year": "1961"
        },
        {
            "id": "doc6",
            "title": "GST on Services - Professional Services",
            "body": "Professional services including legal services, consultancy, accounting, and medical services attract 18% GST under the GST Act. However, certain services are exempt including healthcare services by clinical establishments, educational services by recognized institutions, and services by advocates to clients in relation to any case in any court, tribunal, or authority. Professional services with annual turnover below Rs. 20 lakhs (Rs. 10 lakhs for special category states) are exempt from GST registration. Composition scheme is not available for service providers.",
            "law_type": "GST Act",
            "section": "Schedule II",
            "year": "2017"
        },
        {
            "id": "doc7",
            "title": "Property Transfer Tax and Stamp Duty",
            "body": "Property transfer in India involves stamp duty payment as per the Indian Stamp Act, 1899. Stamp duty rates vary by state: Maharashtra charges 5-6%, Delhi charges 6%, Karnataka charges 5.6%, and Tamil Nadu charges 7%. Registration fees are typically 1% of property value across most states. Circle rate or guidance value is used for stamp duty calculation. Under-reporting of property value may lead to penalties. Some states offer concessions for women buyers, first-time buyers, or affordable housing transactions.",
            "law_type": "Property Law",
            "section": "Indian Stamp Act",
            "year": "1899"
        },
        {
            "id": "doc8",
            "title": "Supreme Court Civil Appeal Procedures",
            "body": "Civil appeals to the Supreme Court of India are governed by the Supreme Court Rules, 2013. Appeals must be filed within 90 days of the judgment/order with certified copies of the judgment, orders, and pleadings. Court fee for civil appeals is Rs. 25,000 plus copying charges. Special Leave Petition (SLP) fee is Rs. 10,000. The appeal must raise substantial questions of law of general public importance. Appearance through advocate-on-record is mandatory. The Supreme Court may grant leave to appeal in cases involving constitutional interpretation or matters of public importance.",
            "law_type": "Supreme Court Rules",
            "section": "Order XLI",
            "year": "2013"
        }
    ]

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    """Extract text from uploaded Word document"""
    doc = Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_entities(text):
    """Extract important terms from text using simple NLP"""
    # Simple entity extraction without spaCy
    # Remove common stop words and extract meaningful terms
    stop_words = {
        'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 'a', 'an', 
        'as', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 
        'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must', 'shall',
        'to', 'of', 'for', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over', 'within'
    }
    
    # Extract words that are likely to be important entities
    entities = set()
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    for word in words:
        if (len(word) > 2 and 
            word not in stop_words and 
            not word.isdigit()):
            entities.add(word)
    
    return entities

def create_embeddings(documents):
    """Create embeddings for all documents"""
    for doc in documents:
        if 'embedding' not in doc:
            doc['embedding'] = sentence_model.encode(doc['body'])
            doc['entities'] = extract_entities(doc['body'])
    return documents

def cosine_similarity_ranking(query_embedding, documents):
    """Rank documents using cosine similarity"""
    scores = []
    for doc in documents:
        score = float(cosine_similarity([query_embedding], [doc['embedding']])[0][0])
        scores.append((doc, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def euclidean_distance_ranking(query_embedding, documents):
    """Rank documents using euclidean distance (negative for ranking)"""
    scores = []
    for doc in documents:
        distance = float(euclidean_distances([query_embedding], [doc['embedding']])[0][0])
        scores.append((doc, -distance))  # Negative for reverse ranking
    return sorted(scores, key=lambda x: x[1], reverse=True)

def mmr_ranking(query_embedding, documents, lambda_param=0.7, top_k=5):
    """Rank documents using Maximal Marginal Relevance"""
    selected = []
    candidates = documents.copy()
    
    while len(selected) < top_k and candidates:
        mmr_scores = []
        for doc in candidates:
            # Relevance score
            relevance = float(cosine_similarity([query_embedding], [doc['embedding']])[0][0])
            
            # Redundancy score (max similarity with already selected documents)
            if selected:
                redundancy = max([
                    float(cosine_similarity([doc['embedding']], [sel_doc['embedding']])[0][0]) 
                    for sel_doc in selected
                ])
            else:
                redundancy = 0
            
            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((doc, mmr_score))
        
        # Select document with highest MMR score
        best_doc = max(mmr_scores, key=lambda x: x[1])
        selected.append(best_doc[0])
        candidates.remove(best_doc[0])
    
    # Return with original relevance scores for comparison
    return [(doc, float(cosine_similarity([query_embedding], [doc['embedding']])[0][0])) for doc in selected]

def hybrid_similarity_ranking(query_embedding, query_entities, documents, w_cosine=0.6, w_entity=0.4):
    """Rank documents using hybrid similarity (cosine + entity matching)"""
    scores = []
    for doc in documents:
        # Cosine similarity score
        cosine_score = float(cosine_similarity([query_embedding], [doc['embedding']])[0][0])
        
        # Entity overlap score
        doc_entities = doc.get('entities', set())
        if len(query_entities) > 0 and len(doc_entities) > 0:
            intersection = len(query_entities.intersection(doc_entities))
            union = len(query_entities.union(doc_entities))
            entity_score = intersection / union if union > 0 else 0
        else:
            entity_score = 0
        
        # Hybrid score
        hybrid_score = float(w_cosine * cosine_score + w_entity * entity_score)
        scores.append((doc, hybrid_score))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)

def calculate_precision_at_k(ranked_docs, relevant_doc_ids, k=5):
    """Calculate precision@k metric"""
    top_k_ids = [doc['id'] for doc, _ in ranked_docs[:k]]
    relevant_retrieved = len(set(top_k_ids).intersection(set(relevant_doc_ids)))
    return relevant_retrieved / k if k > 0 else 0

def calculate_diversity_score(ranked_docs, k=5):
    """Calculate diversity score based on law types"""
    top_k_types = [doc['law_type'] for doc, _ in ranked_docs[:k]]
    unique_types = len(set(top_k_types))
    return unique_types / k if k > 0 else 0

def display_results(method_name, ranked_docs, show_count=5):
    """Display ranked results in a formatted way"""
    st.subheader(f"📊 {method_name}")
    
    for i, (doc, score) in enumerate(ranked_docs[:show_count], 1):
        with st.expander(f"{i}. {doc['title']} (Score: {score:.4f})"):
            st.write(f"**Law Type:** {doc['law_type']}")
            if 'section' in doc:
                st.write(f"**Section:** {doc['section']}")
            if 'year' in doc:
                st.write(f"**Year:** {doc['year']}")
            st.write(f"**Content:** {doc['body'][:200]}...")
            
            # Score visualization
            try:
                progress_value = float(min(max(score, 0), 1))
                if not np.isnan(progress_value) and not np.isinf(progress_value):
                    st.progress(progress_value)
                else:
                    st.progress(0.0)
            except (ValueError, TypeError):
                st.progress(0.0)

def main():
    st.title("⚖️ Indian Legal Document Search System")
    st.markdown("Compare 4 different similarity methods for legal document retrieval")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load sample documents
    documents = get_sample_documents()
    documents = create_embeddings(documents)
    
    # Document upload section
    st.sidebar.subheader("📄 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or Word documents", 
        accept_multiple_files=True, 
        type=['pdf', 'docx']
    )
    
    # Process uploaded files
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(uploaded_file)
            else:
                continue
            
            # Add to documents
            new_doc = {
                "id": f"uploaded_{len(documents)}",
                "title": uploaded_file.name,
                "body": text[:1000],  # Limit text for demo
                "law_type": "Uploaded Document",
                "section": "N/A",
                "year": "N/A"
            }
            documents.append(new_doc)
        
        # Recreate embeddings for new documents
        documents = create_embeddings(documents)
        st.sidebar.success(f"✅ {len(uploaded_files)} documents uploaded successfully!")
    
    # Algorithm parameters
    st.sidebar.subheader("🔧 Algorithm Parameters")
    mmr_lambda = st.sidebar.slider("MMR Lambda (Relevance vs Diversity)", 0.0, 1.0, 0.7, 0.1)
    hybrid_cosine_weight = st.sidebar.slider("Hybrid Cosine Weight", 0.0, 1.0, 0.6, 0.1)
    hybrid_entity_weight = 1.0 - hybrid_cosine_weight
    
    # Test queries
    st.sidebar.subheader("🧪 Predefined Test Queries")
    test_queries = [
        "Income tax deduction for education",
        "GST rate for textile products", 
        "Property registration process",
        "Court fee structure"
    ]
    
    selected_test_query = st.sidebar.selectbox("Select a test query:", [""] + test_queries)
    
    # Main query input
    st.subheader("🔍 Search Query")
    query = st.text_input("Enter your legal query:", value=selected_test_query)
    
    if query:
        # Process query
        query_embedding = sentence_model.encode(query)
        query_entities = extract_entities(query)
        
        # Run all similarity methods
        with st.spinner("🔄 Processing query with all similarity methods..."):
            cosine_results = cosine_similarity_ranking(query_embedding, documents)
            euclidean_results = euclidean_distance_ranking(query_embedding, documents)
            mmr_results = mmr_ranking(query_embedding, documents, mmr_lambda)
            hybrid_results = hybrid_similarity_ranking(
                query_embedding, query_entities, documents, 
                hybrid_cosine_weight, hybrid_entity_weight
            )
        
        # Display results in 4 columns
        st.subheader("📊 Comparison Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            display_results("Cosine Similarity", cosine_results)
            
        with col2:
            display_results("Euclidean Distance", euclidean_results)
            
        with col3:
            display_results("MMR", mmr_results)
            
        with col4:
            display_results("Hybrid Similarity", hybrid_results)
        
        # Performance metrics
        st.subheader("📈 Performance Metrics")
        
        # For demonstration, assume first result is always relevant
        relevant_doc_ids = [result[0]['id'] for result in cosine_results[:2]]  # Top 2 from cosine as "relevant"
        
        # Calculate metrics
        metrics_data = {
            'Method': ['Cosine Similarity', 'Euclidean Distance', 'MMR', 'Hybrid Similarity'],
            'Precision@5': [
                calculate_precision_at_k(cosine_results, relevant_doc_ids, 5),
                calculate_precision_at_k(euclidean_results, relevant_doc_ids, 5),
                calculate_precision_at_k(mmr_results, relevant_doc_ids, 5),
                calculate_precision_at_k(hybrid_results, relevant_doc_ids, 5)
            ],
            'Diversity Score': [
                calculate_diversity_score(cosine_results, 5),
                calculate_diversity_score(euclidean_results, 5),
                calculate_diversity_score(mmr_results, 5),
                calculate_diversity_score(hybrid_results, 5)
            ],
            'Top Score': [
                cosine_results[0][1] if cosine_results else 0,
                euclidean_results[0][1] if euclidean_results else 0,
                mmr_results[0][1] if mmr_results else 0,
                hybrid_results[0][1] if hybrid_results else 0
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Display metrics table
        st.dataframe(metrics_df, use_container_width=True)
        
        # Metrics visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_precision = px.bar(
                metrics_df, 
                x='Method', 
                y='Precision@5',
                title='Precision@5 Comparison',
                color='Precision@5',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_precision, use_container_width=True)
        
        with col2:
            fig_diversity = px.bar(
                metrics_df, 
                x='Method', 
                y='Diversity Score',
                title='Diversity Score Comparison',
                color='Diversity Score',
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig_diversity, use_container_width=True)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        best_precision = metrics_df.loc[metrics_df['Precision@5'].idxmax()]
        best_diversity = metrics_df.loc[metrics_df['Diversity Score'].idxmax()]
        
        st.success(f"**Best Precision:** {best_precision['Method']} ({best_precision['Precision@5']:.3f})")
        st.info(f"**Best Diversity:** {best_diversity['Method']} ({best_diversity['Diversity Score']:.3f})")
        
        # Analysis
        st.markdown("""
        ### 📝 Analysis Notes:
        - **Cosine Similarity**: Standard semantic matching, good for general queries
        - **Euclidean Distance**: Alternative distance metric, may capture different patterns
        - **MMR**: Reduces redundancy, promotes diverse results across different law types
        - **Hybrid Similarity**: Combines semantic similarity with legal entity matching
        """)

if __name__ == "__main__":
    main() 