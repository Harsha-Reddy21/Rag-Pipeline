import numpy as np
from typing import Dict, List, Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import re

def predict_conversion(conversation_text: str, model, customer_context: str = "") -> Dict[str, Any]:
    """Predict conversion probability for a sales conversation."""
    
    # Generate embedding for the conversation
    text_embedding = model.encode([conversation_text])
    
    # Calculate features
    features = extract_prediction_features(conversation_text, customer_context)
    
    # Simulate prediction (in practice, you'd use a trained classifier)
    probability = simulate_conversion_prediction(conversation_text, features)
    
    # Generate insights
    insights = generate_conversation_insights(conversation_text, features)
    
    return {
        'probability': probability,
        'confidence': min(0.95, probability + 0.1),  # Simulated confidence
        'features': features,
        'insights': insights,
        'embedding': text_embedding[0]
    }

def simulate_conversion_prediction(conversation_text: str, features: Dict) -> float:
    """Simulate conversion prediction based on conversation patterns."""
    
    # Positive signals
    positive_keywords = [
        'interested', 'perfect', 'great', 'excellent', 'yes', 'proceed', 
        'forward', 'budget approved', 'ready', 'start', 'implement'
    ]
    
    # Negative signals
    negative_keywords = [
        'expensive', 'costly', 'budget', 'think about', 'maybe', 'later', 
        'compare', 'unsure', 'not sure', 'discuss', 'team decision'
    ]
    
    # Buying intent signals
    buying_signals = [
        'when can we start', 'next steps', 'contract', 'timeline', 
        'implementation', 'onboarding', 'pricing', 'payment'
    ]
    
    text_lower = conversation_text.lower()
    
    # Calculate signal scores
    positive_score = sum(1 for keyword in positive_keywords if keyword in text_lower)
    negative_score = sum(1 for keyword in negative_keywords if keyword in text_lower)
    buying_score = sum(1 for signal in buying_signals if signal in text_lower)
    
    # Base probability calculation
    base_prob = 0.3  # Base conversion rate
    
    # Adjust based on signals
    positive_boost = min(0.4, positive_score * 0.1)
    negative_penalty = min(0.3, negative_score * 0.08)
    buying_boost = min(0.3, buying_score * 0.15)
    
    # Length factor (longer conversations often indicate more engagement)
    length_factor = min(0.1, len(conversation_text.split()) / 1000)
    
    # Final probability
    probability = base_prob + positive_boost - negative_penalty + buying_boost + length_factor
    
    return max(0.05, min(0.95, probability))

def extract_prediction_features(conversation_text: str, customer_context: str = "") -> Dict:
    """Extract features for conversion prediction."""
    
    features = {}
    
    # Text-based features
    words = conversation_text.split()
    sentences = re.split(r'[.!?]+', conversation_text)
    
    features['word_count'] = len(words)
    features['sentence_count'] = len(sentences)
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    
    # Conversation structure
    customer_lines = len(re.findall(r'Customer:', conversation_text))
    rep_lines = len(re.findall(r'(Sales Rep|Rep):', conversation_text))
    
    features['customer_participation'] = customer_lines / max(1, customer_lines + rep_lines)
    features['total_exchanges'] = customer_lines + rep_lines
    
    # Question analysis
    customer_questions = len(re.findall(r'Customer:.*?\?', conversation_text))
    rep_questions = len(re.findall(r'(Sales Rep|Rep):.*?\?', conversation_text))
    
    features['customer_questions'] = customer_questions
    features['rep_questions'] = rep_questions
    features['question_ratio'] = customer_questions / max(1, rep_questions)
    
    # Sentiment indicators
    positive_words = ['great', 'excellent', 'perfect', 'interested', 'yes', 'good']
    negative_words = ['expensive', 'concern', 'problem', 'issue', 'no', 'difficult']
    
    features['positive_sentiment'] = sum(1 for word in positive_words if word in conversation_text.lower())
    features['negative_sentiment'] = sum(1 for word in negative_words if word in conversation_text.lower())
    
    # Context features
    if customer_context:
        features['has_context'] = 1
        features['context_length'] = len(customer_context.split())
    else:
        features['has_context'] = 0
        features['context_length'] = 0
    
    return features

def generate_conversation_insights(conversation_text: str, features: Dict) -> List[str]:
    """Generate insights about the conversation for sales reps."""
    
    insights = []
    
    # Engagement insights
    if features['customer_participation'] > 0.4:
        insights.append("🎯 High customer engagement - actively participating in conversation")
    elif features['customer_participation'] < 0.2:
        insights.append("⚠️ Low customer engagement - may need to ask more engaging questions")
    
    # Question patterns
    if features['customer_questions'] > 3:
        insights.append("❓ Customer asking many questions - shows genuine interest")
    elif features['customer_questions'] == 0:
        insights.append("🤔 Customer not asking questions - may need to encourage interaction")
    
    # Sentiment analysis
    if features['positive_sentiment'] > features['negative_sentiment']:
        insights.append("😊 Overall positive sentiment detected")
    elif features['negative_sentiment'] > 2:
        insights.append("😟 Multiple concerns raised - address objections carefully")
    
    # Conversation length
    if features['word_count'] > 500:
        insights.append("📈 Extended conversation - indicates serious consideration")
    elif features['word_count'] < 100:
        insights.append("⏱️ Brief conversation - may need more discovery")
    
    # Buying signals
    buying_keywords = ['budget', 'timeline', 'start', 'implement', 'contract', 'next steps']
    buying_signals = sum(1 for keyword in buying_keywords if keyword in conversation_text.lower())
    
    if buying_signals >= 2:
        insights.append("🚀 Strong buying signals detected - ready to move forward")
    elif buying_signals == 1:
        insights.append("💡 Some buying interest shown - explore decision criteria")
    else:
        insights.append("🔍 Focus on identifying pain points and decision process")
    
    return insights

def calculate_similarity_scores(target_conversation: str, reference_conversations: List[str], model) -> List[float]:
    """Calculate similarity scores between target and reference conversations."""
    
    # Generate embeddings
    target_embedding = model.encode([target_conversation])
    reference_embeddings = model.encode(reference_conversations)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(target_embedding, reference_embeddings)[0]
    
    return similarities.tolist()

def rank_prospects(conversations: List[Dict], model) -> List[Dict]:
    """Rank prospects by conversion probability."""
    
    scored_conversations = []
    
    for conv in conversations:
        prediction = predict_conversion(conv['transcript'], model, conv.get('customer_context', ''))
        
        scored_conv = conv.copy()
        scored_conv.update({
            'conversion_probability': prediction['probability'],
            'confidence': prediction['confidence'],
            'priority_score': prediction['probability'] * prediction['confidence']
        })
        
        scored_conversations.append(scored_conv)
    
    # Sort by priority score (descending)
    return sorted(scored_conversations, key=lambda x: x['priority_score'], reverse=True) 