import streamlit as st
import numpy as np
from sklearn.cluster import DBSCAN
from PIL import Image

# 1. page settings
st.set_page_config(
    page_title="DBSCANvas Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. style constants
CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1 {font-family: 'Helvetica Neue', sans-serif; font-weight: 700; letter-spacing: -1px;}
    
    .color-card {
        background-color: #ffffff; border-radius: 12px; padding: 15px; margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #f0f0f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .color-card:hover {transform: translateY(-5px); box-shadow: 0 10px 15px rgba(0,0,0,0.1);}
    .color-preview {width: 100%; height: 80px; border-radius: 8px; margin-bottom: 10px; border: 1px solid rgba(0,0,0,0.05);}
    .hex-code {font-family: 'SF Mono', monospace; font-weight: 600; color: #333; font-size: 1.1rem;}
    .percentage {color: #888; font-size: 0.9rem;}
    
    @media (prefers-color-scheme: dark) {
        .color-card {background-color: #262730; border: 1px solid #363945;}
        .hex-code {color: #fff;}
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. core algorithm
@st.cache_data
def run_dbscan(pixels, eps, min_samples):
    """Run DBSCAN and return sorted color clusters."""
    # core clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pixels)
    labels = db.labels_
    
    # count results
    unique_labels = set(labels) - {-1} # remove noise by set subtraction
    results = []
    
    for k in unique_labels:
        mask = (labels == k)
        count = np.sum(mask)
        avg_color = pixels[mask].mean(axis=0)
        results.append((count, avg_color))
    
    # sort by count
    return sorted(results, key=lambda x: x[0], reverse=True)

# 4. sidebar settings
st.sidebar.title("🎛️ Control Panel")
eps = st.sidebar.slider("🎨 Color Tolerance (EPS)", 0.01, 0.30, 0.08, 0.01)
min_samples = st.sidebar.slider("🔍 Min Samples", 10, 500, 60)

st.sidebar.markdown("---")
resize_size = st.sidebar.slider("📐 Processing Size", 100, 300, 150, 10)
min_percentage = st.sidebar.slider("🎯 Min Show %", 0.1, 5.0, 0.5, 0.1)
cols_per_row = st.sidebar.selectbox("📊 Grid Columns", [2, 3, 4], index=1)

# 5. main interface logic
st.title("DBSCANvas")
st.markdown("##### 🤖 Intelligent Color Extraction System")

uploaded_file = st.file_uploader("📂 Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    try:
        # read and preprocess
        image = Image.open(uploaded_file).convert('RGB')
        
        # layout: left image, right results
        col_img, col_res = st.columns([1.2, 2])
        
        with col_img:
            st.image(image, caption=f"Original ({image.size[0]}x{image.size[1]})", use_container_width=True)

        with col_res:
            with st.spinner('Calculating...'):
                # key optimization: resize first, then pass to cache function, greatly reduce memory overhead
                img_small = image.resize((resize_size, resize_size))
                pixels = np.array(img_small).reshape(-1, 3) / 255.0
                
                # call cache function
                color_counts = run_dbscan(pixels, eps, min_samples)
                total_pixels = len(pixels)

                if color_counts:
                    st.markdown(f"### ✨ Found {len(color_counts)} Colors")
                    grid = st.columns(cols_per_row)
                    
                    valid_count = 0
                    for count, color in color_counts:
                        pct = (count / total_pixels) * 100
                        if pct < min_percentage: continue
                        
                        r, g, b = (int(c * 255) for c in color)
                        hex_c = '#{:02x}{:02x}{:02x}'.format(r, g, b)
                        
                        # render cards
                        with grid[valid_count % cols_per_row]:
                            st.markdown(f"""
                            <div class="color-card">
                                <div class="color-preview" style="background-color: {hex_c};"></div>
                                <div class="hex-code">{hex_c.upper()}</div>
                                <div class="percentage">RGB({r},{g},{b})</div>
                                <div class="percentage">{pct:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        valid_count += 1
                        
                    if valid_count == 0:
                        st.warning("Colors found but hidden due to 'Min Show %' setting.")
                else:
                    st.error("⚠️ No clusters found. Try lowering Min Samples.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👈 Upload an image to start.")