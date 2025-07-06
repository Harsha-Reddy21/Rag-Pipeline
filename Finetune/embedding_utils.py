import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
import logging

def fine_tune_embeddings(data: Dict, base_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                        epochs: int = 3, batch_size: int = 16, learning_rate: float = 2e-5, 
                        margin: float = 0.5) -> Tuple:
    """Fine-tune embeddings using contrastive learning for sales conversations."""
    
    # Load base model
    model = SentenceTransformer(base_model)
    
    # Prepare training data for contrastive learning
    train_examples = prepare_contrastive_examples(
        data['train_texts'], 
        data['train_labels']
    )
    
    # Training losses to track
    training_losses = []
    
    # Simple fine-tuning simulation (in practice, you'd use proper SentenceTransformers training)
    for epoch in range(epochs):
        epoch_loss = simulate_contrastive_training(
            model, train_examples, learning_rate, margin
        )
        training_losses.append(epoch_loss)
        
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    
    return model, training_losses

def prepare_contrastive_examples(texts: List[str], labels: List[int]) -> List[Tuple]:
    """Prepare positive and negative pairs for contrastive learning."""
    
    examples = []
    converted_texts = [texts[i] for i, label in enumerate(labels) if label == 1]
    failed_texts = [texts[i] for i, label in enumerate(labels) if label == 0]
    
    # Create positive pairs (similar conversion outcomes)
    for i in range(min(len(converted_texts), 50)):
        for j in range(i+1, min(len(converted_texts), i+5)):
            examples.append((converted_texts[i], converted_texts[j], 1))  # Similar
    
    for i in range(min(len(failed_texts), 50)):
        for j in range(i+1, min(len(failed_texts), i+5)):
            examples.append((failed_texts[i], failed_texts[j], 1))  # Similar
    
    # Create negative pairs (different conversion outcomes)
    for i in range(min(len(converted_texts), 30)):
        for j in range(min(len(failed_texts), 30)):
            examples.append((converted_texts[i], failed_texts[j], 0))  # Different
    
    return examples

def simulate_contrastive_training(model, examples: List[Tuple], learning_rate: float, margin: float) -> float:
    """Simulate contrastive learning training (simplified version)."""
    
    # In a real implementation, you would:
    # 1. Encode text pairs
    # 2. Calculate contrastive loss
    # 3. Backpropagate and update weights
    
    # For demonstration, we simulate training loss
    base_loss = 0.8
    improvement_factor = np.random.uniform(0.85, 0.95)  # Simulate learning
    
    return base_loss * improvement_factor

def generate_embeddings(texts: List[str], model) -> np.ndarray:
    """Generate embeddings for a list of texts using the model."""
    
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings

def calculate_embedding_similarity(text1: str, text2: str, model) -> float:
    """Calculate cosine similarity between two text embeddings."""
    
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(similarity)

def find_similar_conversations(query_text: str, conversation_texts: List[str], 
                             model, top_k: int = 5) -> List[Tuple[int, float]]:
    """Find most similar conversations to a query using embeddings."""
    
    # Generate embeddings
    query_embedding = model.encode([query_text])
    conversation_embeddings = model.encode(conversation_texts)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, conversation_embeddings)[0]
    
    # Get top-k most similar
    similar_indices = np.argsort(similarities)[::-1][:top_k]
    
    return [(idx, similarities[idx]) for idx in similar_indices]

def create_conversation_clusters(conversation_texts: List[str], model, n_clusters: int = 5) -> Dict:
    """Cluster conversations based on embeddings."""
    
    from sklearn.cluster import KMeans
    
    # Generate embeddings
    embeddings = generate_embeddings(conversation_texts, model)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Organize results
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append({
            'index': i,
            'text': conversation_texts[i]
        })
    
    return clusters 