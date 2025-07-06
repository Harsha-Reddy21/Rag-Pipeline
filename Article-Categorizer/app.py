import streamlit as st
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from models import load_models, train_models, classify_text

# Page config
st.set_page_config(page_title="Smart Article Categorizer", page_icon="📰")

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

def main():
    st.title("📰 Smart Article Categorizer")
    st.markdown("Compare 4 different embedding approaches for article classification")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # OpenAI API Key
        if not os.getenv('OPENAI_API_KEY'):
            api_key = st.text_input("OpenAI API Key (optional):", type="password")
            if api_key:
                os.environ['OPENAI_API_KEY'] = api_key
                st.success("API Key set!")
        else:
            st.success("✅ OpenAI API Key found")
    
    # Main content
    tab1, tab2 = st.tabs(["🔬 Train Models", "🎯 Classify Articles"])
    
    with tab1:
        train_models_tab()
    
    with tab2:
        classify_articles_tab()

def train_models_tab():
    st.header("🔬 Train Models")
    
    # Load models button
    if not st.session_state.models_loaded:
        if st.button("🚀 Load Embedding Models"):
            with st.spinner("Loading models..."):
                load_models()
                st.session_state.models_loaded = True
            st.success("Models loaded! Now you can train them.")
            st.rerun()
    
    # Upload dataset
    if st.session_state.models_loaded:
        st.subheader("📁 Dataset")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        elif os.path.exists("news_dataset.csv"):
            if st.button("📊 Use Sample Dataset"):
                df = pd.read_csv("news_dataset.csv")
                st.success("Sample dataset loaded!")
            else:
                df = None
        else:
            df = None
        
        # Show dataset info
        if 'df' in locals() and df is not None:
            st.write(f"**Dataset shape:** {df.shape}")
            st.write(f"**Categories:** {df['label'].unique().tolist()}")
            st.dataframe(df.head())
            
            # Train models
            if st.button("🎯 Train All Models"):
                with st.spinner("Training models... This may take a few minutes."):
                    try:
                        results = train_models(df)
                        st.session_state.models_trained = True
                        
                        st.success("🎉 Training completed!")
                        
                        # Show results
                        st.subheader("📊 Results")
                        for model, accuracy in results.items():
                            st.metric(f"{model.upper()}", f"{accuracy:.3f}")
                        
                    except Exception as e:
                        st.error(f"Training failed: {e}")

def classify_articles_tab():
    st.header("🎯 Classify Articles")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the 'Train Models' tab.")
        return
    
    # Sample articles
    samples = {
        "Technology": "Apple unveiled its latest iPhone with advanced AI capabilities and improved camera technology.",
        "Finance": "The Federal Reserve announced a 0.25% interest rate cut to stimulate economic growth.",
        "Healthcare": "Researchers at Johns Hopkins have developed a new treatment for Alzheimer's disease.",
        "Sports": "The Lakers defeated the Warriors 112-108 in a thrilling NBA game last night.",
        "Politics": "The Senate passed a bipartisan infrastructure bill with overwhelming support.",
        "Entertainment": "The highly anticipated Marvel movie broke box office records this weekend."
    }
    
    # Input options
    input_type = st.radio("Choose input:", ["✍️ Type your text", "📄 Use sample article"])
    
    if input_type == "✍️ Type your text":
        text = st.text_area("Enter article text:", height=150)
    else:
        selected_sample = st.selectbox("Select sample:", list(samples.keys()))
        text = samples[selected_sample]
        st.text_area("Sample article:", value=text, height=150, disabled=True)
    
    # Classify button
    if text and st.button("🚀 Classify Article"):
        with st.spinner("Classifying..."):
            results = classify_text(text)
            
        if isinstance(results, str):  # Error message
            st.error(results)
        else:
            st.subheader("🎯 Results")
            
            # Create columns for results
            cols = st.columns(len(results))
            
            for i, (model, result) in enumerate(results.items()):
                with cols[i]:
                    st.metric(
                        label=f"**{model.upper()}**",
                        value=result['category'],
                        delta=f"{result['confidence']:.2f} confidence"
                    )
            
            # Show detailed results
            with st.expander("📊 Detailed Results"):
                for model, result in results.items():
                    st.write(f"**{model.upper()}:**")
                    st.write(f"- Category: {result['category']}")
                    st.write(f"- Confidence: {result['confidence']:.3f}")
                    st.write("---")

if __name__ == "__main__":
    main() 