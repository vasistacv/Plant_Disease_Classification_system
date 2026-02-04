import streamlit as st
from PIL import Image
import os
from inference_pipeline_torch import ArcanutSystem # Assuming this is the name or I rename it
import sys

# Add src to path if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_pipeline_torch import ArcanutSystem

# Page Config
st.set_page_config(page_title="Arecanut Disease Doctor", page_icon="🌴", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        width: 100%;
    }
    .report {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    return ArcanutSystem()

st.title("🌴 Arecanut Doctor AI")
st.markdown("### Intelligent Disease Diagnosis System")

# Load System (Lazy loading)
with st.spinner("Initializing AI System (GPU Enabled)..."):
    system = load_system()
    # Trigger load
    system.load_models()

st.success(f"System Ready on {system.device}")

uploaded_file = st.file_uploader("Upload Leaf/Nut Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save temp
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(uploaded_file, caption="Analysis Image", use_column_width=True)
        
    with col2:
        if st.button("Diagnose Plant"):
            with st.spinner("Analyzing textures and patterns..."):
                result = system.predict("temp.jpg")
                
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("### Results")
                    
                    # Identification
                    if result['is_arecanut']:
                        st.success(result['message'])
                        
                        # Disease
                        d_name = result.get('disease')
                        d_conf = result.get('disease_confidence', 0.0) * 100
                        
                        st.markdown(f"""
                        <div class="report">
                            <h2>Diagnosis: {d_name}</h2>
                            <p>Confidence: <strong>{d_conf:.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.progress(int(d_conf))
                        
                        # Recommendations
                        st.markdown("#### Treatment Plan")
                        if d_name == "Healthy":
                            st.info("Plant is healthy. Validated by AI.")
                        elif d_name == "Mahali_Koleroga":
                            st.warning("Rx: 1% Bordeaux Mixture spray immediately.")
                        elif "Yellow" in str(d_name):
                            st.warning("Rx: Improve soil drainage and add Potash.")
                        else:
                            st.warning("Rx: Consult local expert for specific fungicide.")
                            
                    else:
                        st.error(result['message'])
                        st.warning("Analysis stopped. Image does not contain Arecanut signatures.")

    # Cleanup
    # os.remove("temp.jpg") 
