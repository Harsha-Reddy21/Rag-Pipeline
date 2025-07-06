import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from gensim.models import KeyedVectors
    nltk.download('punkt', quiet=True)
    HAS_WORD2VEC = True
except:
    HAS_WORD2VEC = False

try:
    from transformers import BertTokenizer, BertModel
    import torch
    HAS_BERT = True
except:
    HAS_BERT = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except:
    HAS_SBERT = False

try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except:
    HAS_OPENAI = False

# Global variables to store models
word2vec_model = None
bert_tokenizer = None
bert_model = None
sbert_model = None
openai_client = None
label_encoder = None
classifiers = {}

def load_models():
    """Load all available embedding models."""
    global word2vec_model, bert_tokenizer, bert_model, sbert_model, openai_client
    
    # Word2Vec
    if HAS_WORD2VEC:
        try:
            word2vec_model = KeyedVectors.load("word2vec-google-news-300/word2vec-google-news-300.model")
            print("✅ Word2Vec loaded")
        except:
            print("⚠️ Word2Vec not found")
    
    # BERT
    if HAS_BERT:
        try:
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')
            print("✅ BERT loaded")
        except:
            print("⚠️ BERT failed to load")
    
    # Sentence-BERT
    if HAS_SBERT:
        try:
            sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ Sentence-BERT loaded")
        except:
            print("⚠️ Sentence-BERT failed to load")
    
    # OpenAI
    if HAS_OPENAI and os.getenv('OPENAI_API_KEY'):
        try:
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            print("✅ OpenAI client ready")
        except:
            print("⚠️ OpenAI client failed")

def get_word2vec_embedding(text):
    """Get Word2Vec embedding."""
    if not word2vec_model:
        return np.zeros(300)
    
    tokens = word_tokenize(text.lower())
    vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

def get_bert_embedding(text):
    """Get BERT embedding."""
    if not bert_tokenizer or not bert_model:
        return np.zeros(768)
    
    tokens = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**tokens)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def get_sbert_embedding(text):
    """Get Sentence-BERT embedding."""
    if not sbert_model:
        return np.zeros(384)
    
    return sbert_model.encode([text])[0]

def get_openai_embedding(text):
    """Get OpenAI embedding."""
    if not openai_client:
        return np.zeros(1536)
    
    try:
        response = openai_client.embeddings.create(model="text-embedding-ada-002", input=text)
        return np.array(response.data[0].embedding)
    except:
        return np.zeros(1536)

def train_models(df):
    """Train all available models."""
    global label_encoder, classifiers
    
    # Prepare data
    df['text'] = df['text'].fillna("")
    label_encoder = LabelEncoder()
    df['label_enc'] = label_encoder.fit_transform(df['label'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label_enc'], test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Train each available model
    models = [
        ('word2vec', get_word2vec_embedding, word2vec_model is not None),
        ('bert', get_bert_embedding, bert_model is not None),
        ('sbert', get_sbert_embedding, sbert_model is not None),
        ('openai', get_openai_embedding, openai_client is not None)
    ]
    
    for name, embedding_func, available in models:
        if not available:
            continue
            
        print(f"Training {name}...")
        
        # Get embeddings
        train_embeddings = np.vstack([embedding_func(text) for text in X_train])
        test_embeddings = np.vstack([embedding_func(text) for text in X_test])
        
        # Train classifier
        clf = LogisticRegression(max_iter=1000)
        clf.fit(train_embeddings, y_train)
        
        # Evaluate
        y_pred = clf.predict(test_embeddings)
        accuracy = accuracy_score(y_test, y_pred)
        
        classifiers[name] = clf
        results[name] = accuracy
        
        print(f"✅ {name}: {accuracy:.3f}")
    
    return results

def classify_text(text):
    """Classify text using all trained models."""
    if not classifiers:
        return "Please train models first!"
    
    results = {}
    embedding_funcs = {
        'word2vec': get_word2vec_embedding,
        'bert': get_bert_embedding,
        'sbert': get_sbert_embedding,
        'openai': get_openai_embedding
    }
    
    for name, clf in classifiers.items():
        embedding = embedding_funcs[name](text)
        prediction = clf.predict([embedding])[0]
        confidence = clf.predict_proba([embedding])[0].max()
        
        results[name] = {
            'category': label_encoder.inverse_transform([prediction])[0],
            'confidence': confidence
        }
    
    return results

 