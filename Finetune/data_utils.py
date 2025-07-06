import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import json

def generate_sample_data(num_samples: int = 200, conversion_rate: float = 0.25) -> pd.DataFrame:
    """Generate sample sales conversation data with conversion labels."""
    np.random.seed(42)
    
    conversations = []
    
    for i in range(num_samples):
        # Randomly determine if this conversation converts
        converts = np.random.random() < conversion_rate
        
        # Generate conversation based on conversion outcome
        if converts:
            transcript = generate_positive_conversation()
            sentiment_score = np.random.normal(0.7, 0.15)
        else:
            transcript = generate_negative_conversation()
            sentiment_score = np.random.normal(0.3, 0.15)
        
        conversations.append({
            'conversation_id': f"conv_{i+1:04d}",
            'transcript': transcript,
            'converted': converts,
            'sentiment_score': max(0, min(1, sentiment_score)),
            'duration_minutes': np.random.normal(25, 8),
            'customer_type': np.random.choice(['enterprise', 'smb', 'startup']),
            'sales_rep': f"rep_{np.random.randint(1, 10)}"
        })
    
    return pd.DataFrame(conversations)

def generate_positive_conversation() -> str:
    """Generate a conversation likely to convert."""
    positive_patterns = [
        "Customer: I'm really interested in your solution. It seems like exactly what we need for our growing business.",
        "Sales Rep: That's great to hear! Let me walk you through how our platform can specifically address your challenges.",
        "Customer: The ROI projections look compelling. When can we start implementation?",
        "Sales Rep: We can begin onboarding within two weeks. Our dedicated success team will ensure smooth deployment.",
        "Customer: Perfect! I'd like to move forward. What are the next steps?",
        "Sales Rep: Excellent! I'll send over the contract today and schedule our kickoff call."
    ]
    
    return "\n".join(positive_patterns)

def generate_negative_conversation() -> str:
    """Generate a conversation unlikely to convert."""
    negative_patterns = [
        "Customer: I'm just browsing and comparing options. Not ready to make any decisions yet.",
        "Sales Rep: I understand. Let me show you what makes our solution unique in the market.",
        "Customer: The price point is concerning. We have a limited budget this quarter.",
        "Sales Rep: We do offer flexible payment plans and can work within your budget constraints.",
        "Customer: I need to discuss this with my team first. We'll get back to you in a few weeks.",
        "Sales Rep: Of course! I'll follow up next month to see how your evaluation is progressing."
    ]
    
    return "\n".join(negative_patterns)

def preprocess_conversations(conversations_df: pd.DataFrame) -> Dict:
    """Preprocess conversation data for training."""
    
    # Clean and tokenize transcripts
    cleaned_transcripts = []
    labels = []
    
    for idx, row in conversations_df.iterrows():
        # Clean transcript
        cleaned_text = clean_text(row['transcript'])
        cleaned_transcripts.append(cleaned_text)
        labels.append(int(row['converted']))
    
    # Split into train/test
    split_idx = int(0.8 * len(cleaned_transcripts))
    
    return {
        'train_texts': cleaned_transcripts[:split_idx],
        'train_labels': labels[:split_idx],
        'test_texts': cleaned_transcripts[split_idx:],
        'test_labels': labels[split_idx:],
        'all_texts': cleaned_transcripts,
        'all_labels': labels
    }

def clean_text(text: str) -> str:
    """Clean and normalize conversation text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize speaker labels
    text = re.sub(r'(Customer|Sales Rep|Rep):', '', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text.strip()

def extract_conversation_features(transcript: str) -> Dict:
    """Extract features from conversation transcript."""
    
    features = {}
    
    # Basic text features
    features['word_count'] = len(transcript.split())
    features['sentence_count'] = len(re.split(r'[.!?]+', transcript))
    features['avg_sentence_length'] = features['word_count'] / max(1, features['sentence_count'])
    
    # Sales-specific features
    positive_keywords = ['interested', 'perfect', 'great', 'excellent', 'yes', 'proceed', 'forward']
    negative_keywords = ['expensive', 'budget', 'think', 'maybe', 'later', 'compare', 'unsure']
    
    features['positive_signals'] = sum(1 for word in positive_keywords if word in transcript.lower())
    features['negative_signals'] = sum(1 for word in negative_keywords if word in transcript.lower())
    features['signal_ratio'] = features['positive_signals'] / max(1, features['negative_signals'])
    
    # Question patterns
    features['customer_questions'] = len(re.findall(r'Customer:.*?\?', transcript))
    features['rep_questions'] = len(re.findall(r'(Sales Rep|Rep):.*?\?', transcript))
    
    return features 