import pandas as pd
import numpy as np
import random
from typing import List, Dict
from datetime import datetime, timedelta

def create_sample_sales_data(num_samples: int = 200, conversion_rate: float = 0.25, 
                           include_metadata: bool = True, add_noise: bool = True) -> pd.DataFrame:
    """Create realistic sample sales conversation data."""
    
    np.random.seed(42)
    random.seed(42)
    
    conversations = []
    
    for i in range(num_samples):
        # Determine conversion outcome
        converts = np.random.random() < conversion_rate
        
        # Generate realistic conversation
        conversation_data = generate_realistic_conversation(i+1, converts, include_metadata, add_noise)
        conversations.append(conversation_data)
    
    return pd.DataFrame(conversations)

def generate_realistic_conversation(conv_id: int, converts: bool, include_metadata: bool, add_noise: bool) -> Dict:
    """Generate a single realistic sales conversation.""" 