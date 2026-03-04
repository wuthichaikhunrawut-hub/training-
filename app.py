"""
NeuroCore - Perceptron Analytics Dashboard
A Futuristic/Gaming Dashboard for Perceptron Neural Network Analysis

Author: MiniMax Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Import custom modules
from data_engine import (
    load_data, clean_missing_values, min_max_scaler,
    get_data_statistics, get_feature_columns, auto_label_encode
)
from perceptron import (
    perceptron_predict, get_activation_curve, get_formula_latex, train_perceptron
)

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="NeuroCore | Perceptron Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Futuristic/Gaming Theme
# ============================================
st.markdown("""
<style>
    /* Import Google Font - Orbitron for futuristic look */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0a12 100%);
        font-family: 'Rajdhani', sans-serif;
    }

    /* Neon Glow Effects */
    .neon-blue {
        color: #00F2FF !important;
        text-shadow: 0 0 10px #00F2FF, 0 0 20px #00F2FF, 0 0 30px #00F2FF;
    }

    .neon-purple {
        color: #BC13FE !important;
        text-shadow: 0 0 10px #BC13FE, 0 0 20px #BC13FE;
    }

    .neon-emerald {
        color: #00FF9F !important;
        text-shadow: 0 0 10px #00FF9F, 0 0 20px #00FF9F;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        font-weight: 700;
        letter-spacing: 2px;
    }

    h1 {
        font-size: 2.5rem !important;
        background: linear-gradient(90deg, #00F2FF, #BC13FE, #00FF9F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
    }

    h2 {
        color: #00F2FF !important;
        border-bottom: 2px solid #BC13FE;
        padding-bottom: 10px;
        margin-top: 30px;
    }

    h3 {
        color: #BC13FE !important;
    }

    /* Custom Container Boxes */
    .cyber-box {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid #BC13FE;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 15px rgba(188, 19, 254, 0.3), inset 0 0 30px rgba(0, 242, 255, 0.05);
        margin: 15px 0;
    }

    .neon-box {
        background: linear-gradient(135deg, rgba(0, 242, 255, 0.1) 0%, rgba(188, 19, 254, 0.1) 100%);
        border: 2px solid #00F2FF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.4), 0 0 40px rgba(0, 242, 255, 0.2);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0a12 100%);
        border-right: 2px solid #BC13FE;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        margin-bottom: 10px;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #00F2FF 0%, #BC13FE 100%);
        color: #0a0a0f !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.5);
    }

    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.8), 0 0 50px rgba(188, 19, 254, 0.5);
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(13, 17, 23, 0.9) !important;
        border: 1px solid #00F2FF !important;
        border-radius: 8px !important;
        color: #E0E0E0 !important;
        font-family: 'Rajdhani', sans-serif;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.5) !important;
        border-color: #BC13FE !important;
    }

    /* Sliders */
    .stSlider > div > div > div[role="slider"] {
        background: linear-gradient(135deg, #00F2FF, #BC13FE) !important;
        box-shadow: 0 0 10px #00F2FF !important;
    }

    /* DataFrame */
    div[data-testid="stDataFrame"] {
        border: 1px solid #BC13FE;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(188, 19, 254, 0.3);
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem !important;
        color: #00FF9F !important;
        text-shadow: 0 0 15px #00FF9F;
    }

    div[data-testid="stMetricLabel"] {
        color: #BC13FE !important;
        font-family: 'Orbitron', sans-serif;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00F2FF, #BC13FE, #00FF9F) !important;
        box-shadow: 0 0 10px #00F2FF;
    }

    /* Notifications */
    div[data-testid="stNotification"] {
        border-radius: 10px;
    }

    /* Tabs */
    .stTabs > div[role="tablist"] > button {
        background: transparent !important;
        color: #E0E0E0 !important;
        border: 1px solid #BC13FE !important;
        border-radius: 8px 8px 0 0 !important;
        font-family: 'Orbitron', sans-serif;
    }

    .stTabs > div[role="tablist"] > button[aria-selected="true"] {
        background: rgba(188, 19, 254, 0.3) !important;
        color: #00F2FF !important;
        border-bottom: 2px solid #00FF9F !important;
    }

    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00F2FF, #BC13FE, #00FF9F, transparent);
        margin: 30px 0;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0d1117;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00F2FF, #BC13FE);
        border-radius: 4px;
    }

    /* Training table highlight */
    .training-table-section {
        background: rgba(0, 242, 255, 0.03);
        border: 1px solid #00F2FF;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Lottie Animation Functions
# ============================================
def load_lottieurl(url):
    """Load Lottie animation from URL"""
    try:
        import requests
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


# Lottie Animation URLs (Free to use)
LOTTIE_AI = "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json"
LOTTIE_DATA = "https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json"
LOTTIE_TECH = "https://assets2.lottiefiles.com/packages/lf20_w51pcehl.json"


# ============================================
# Session State Initialization
# ============================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'df_encoded' not in st.session_state:
    st.session_state.df_encoded = None
if 'df_scaled' not in st.session_state:
    st.session_state.df_scaled = None
if 'mappings' not in st.session_state:
    st.session_state.mappings = {}
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'weights' not in st.session_state:
    st.session_state.weights = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'epochs' not in st.session_state:
    st.session_state.epochs = 10
if 'calculation_done' not in st.session_state:
    st.session_state.calculation_done = False
if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'training_history' not in st.session_state:
    st.session_state.training_history = None
if 'epoch_errors' not in st.session_state:
    st.session_state.epoch_errors = []
if 'final_weights' not in st.session_state:
    st.session_state.final_weights = None
if 'final_threshold' not in st.session_state:
    st.session_state.final_threshold = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.0
if 'learning_rate' not in st.session_state:
    st.session_state.learning_rate = 0.2
if 'activation_internal' not in st.session_state:
    st.session_state.activation_internal = 'threshold'
if 'gain' not in st.session_state:
    st.session_state.gain = 1.0


# ============================================
# Sidebar - Configuration Panel
# ============================================
with st.sidebar:
    # Logo/Title
    st.markdown("""
    <div style="text-align: center; padding: 10px;">
        <h1 class="neon-blue">🧠 NEUROCORE</h1>
        <p style="color: #BC13FE; font-family: 'Orbitron';">PERCEPTRON ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)

    # Lottie Animation - AI Theme
    try:
        from streamlit_lottie import st_lottie
        lottie_ai = load_lottieurl(LOTTIE_AI)
        if lottie_ai:
            st_lottie(lottie_ai, height=150, key="ai_anim")
    except:
        st.markdown("🤖")

    st.markdown("---")

    # File Upload Section
    st.markdown('<h2 class="neon-purple">📂 Data Input</h2>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV/ARFF)",
        type=['csv', 'arff'],
        help="Upload your training data in CSV or ARFF format"
    )

    # Neural Network Configuration
    st.markdown("---")
    st.markdown('<h2 class="neon-purple">⚙️ Neural Config</h2>', unsafe_allow_html=True)

    # Activation Function Selection
    activation_type = st.selectbox(
        "เลือกสมการ (Activation Function)",
        ["Threshold (Hard-limit)", "Sigmoid"],
        help="เลือกฟังก์ชันการทำงานของ Perceptron"
    )

    # Convert to internal type
    activation_internal = 'threshold' if activation_type == 'Threshold (Hard-limit)' else 'sigmoid'

    # Gain parameter (only for sigmoid)
    gain = 1.0
    if activation_internal == 'sigmoid':
        gain = st.number_input(
            "Gain (a)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            format="%.4f",
            help="Sigmoid gain parameter. Higher values make the curve steeper."
        )

    # Threshold Number Input — free decimal
    threshold = st.number_input(
        "กำหนดค่า Threshold (θ)",
        value=0.0,
        format="%.4f",
        help="Decision threshold (θ) สำหรับสมการ z = Σ(wᵢxᵢ) − θ"
    )

    # Learning Rate Input — free decimal
    learning_rate = st.number_input(
        "กำหนดค่า Learning Rate (α)",
        value=0.2,
        format="%.4f",
        help="Learning rate (α) สำหรับปรับค่าน้ำหนัก"
    )

    # Store in session state for later use
    st.session_state.threshold = threshold
    st.session_state.learning_rate = learning_rate
    st.session_state.activation_internal = activation_internal
    st.session_state.gain = gain

    st.markdown("---")

    # Training Control Section
    st.markdown('<h2 class="neon-purple">⏳ Training Control</h2>', unsafe_allow_html=True)

    epochs = st.number_input(
        "จำนวนรอบการคำนวณ (Epochs)",
        min_value=1,
        max_value=10000,
        value=10,
        step=1,
        help="จำนวนรอบที่ต้องการให้โมเดลวนซ้ำเพื่อปรับค่าน้ำหนัก (หยุดอัตโนมัติเมื่อไม่มี Error)"
    )

    # Store epochs in session state
    st.session_state.epochs = epochs

    if st.button("🔥 START TRAINING", use_container_width=True, key="training_btn"):
        if not st.session_state.data_loaded:
            st.error("❌ โปรดอัปโหลดข้อมูลก่อน")
        elif not st.session_state.feature_cols:
            st.error("❌ โปรดเลือก Features ก่อน")
        elif not st.session_state.weights:
            st.error("❌ โปรดกำหนดค่า Weights ก่อน")
        else:
            try:
                with st.spinner("🔄 กำลังฝึกโมเดล..."):
                    # Prepare training data
                    X_train = st.session_state.df_scaled[st.session_state.feature_cols].values
                    y_train = st.session_state.df_encoded[st.session_state.target_col].values

                    # Use target column data directly as per Slide 19 requirements
                    y_binary = y_train.flatten()

                    # Run training
                    final_weights, final_threshold, training_history, epoch_errors = train_perceptron(
                        X_train,
                        y_binary,
                        st.session_state.weights,
                        st.session_state.threshold,
                        st.session_state.learning_rate,
                        st.session_state.epochs,
                        st.session_state.activation_internal,
                        st.session_state.gain
                    )

                    # Store results in session state
                    st.session_state.final_weights = final_weights
                    st.session_state.final_threshold = final_threshold
                    st.session_state.training_history = training_history
                    st.session_state.trained_feature_names = st.session_state.feature_cols.copy()
                    st.session_state.epoch_errors = epoch_errors
                    st.session_state.training_completed = True

                st.success("✅ เสร็จสิ้นการฝึกโมเดล!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")

    st.markdown("---")

    # Mathematical Formula Display
    st.markdown('<h3 class="neon-emerald">📐 Perceptron Formula</h3>', unsafe_allow_html=True)

    formula_latex = get_formula_latex(activation_internal, gain)
    st.latex(formula_latex)


# ============================================
# Main Content Area
# ============================================

# Header Section
st.markdown("""
<div class="neon-box" style="text-align: center; margin-bottom: 30px;">
    <h1>🧠 PERCEPTRON NEURAL NETWORK</h1>
    <p style="color: #E0E0E0; font-size: 1.2rem;">
        Interactive Dashboard for Neural Network Analysis &amp; Prediction
    </p>
</div>
""", unsafe_allow_html=True)

# Data Processing Section
if uploaded_file is not None:
    with st.spinner('🔄 Loading and processing data...'):
        # Load data
        df, error = load_data(uploaded_file)

        if error:
            st.error(f"❌ Error loading file: {error}")
        else:
            # Display original data info
            st.markdown(f"""
            <div class="cyber-box">
                <h3 class="neon-blue">📊 Dataset Information</h3>
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            </div>
            """, unsafe_allow_html=True)

            # Auto-cleaning: Missing Values
            df_clean, clean_notification, rows_removed = clean_missing_values(df)

            if clean_notification:
                st.warning(clean_notification)

            # Apply Auto Label Encoding for categorical data
            df_encoded, mappings = auto_label_encode(df_clean)

            # Apply Min-Max Scaling on encoded data
            df_scaled = min_max_scaler(df_encoded)

            # Get feature columns from encoded data
            feature_cols, target_col = get_feature_columns(df_encoded)

            # Store in session state
            st.session_state.data_loaded = True
            st.session_state.df_clean = df_clean
            st.session_state.df_encoded = df_encoded
            st.session_state.df_scaled = df_scaled
            st.session_state.mappings = mappings
            st.session_state.feature_cols = feature_cols
            st.session_state.target_col = target_col
            st.session_state.weights = [0.5] * len(feature_cols)

            # Display processed data
            st.markdown("---")
            st.markdown('<h2 class="neon-blue">📈 Data Preview & Mappings</h2>', unsafe_allow_html=True)
            
            # Display Encoding Mappings if they exist
            if st.session_state.mappings:
                with st.expander("🔗 Categorical Mappings (Text to Numeric)", expanded=True):
                    cols = st.columns(len(st.session_state.mappings) if len(st.session_state.mappings) <= 3 else 3)
                    for i, (col_name, mapping) in enumerate(st.session_state.mappings.items()):
                        with cols[i % len(cols)]:
                            st.write(f"**{col_name}**")
                            map_df = pd.DataFrame([{"Value": v, "Code": k} for k, v in mapping.items()])
                            st.table(map_df)
            else:
                st.info("ℹ️ No categorical columns detected. Data is already numeric.")

            # Tab for different views
            tab1, tab2, tab3 = st.tabs(["🔍 Encoded Data", "📊 Statistics", "⚖️ Scaled Data"])

            with tab1:
                st.dataframe(
                    st.session_state.df_encoded,
                    use_container_width=True,
                    height=300
                )

            with tab2:
                stats = get_data_statistics(st.session_state.df_clean)
                if stats is not None:
                    st.dataframe(
                        stats[['count', 'mean', 'std', 'min', 'max', 'range']],
                        use_container_width=True
                    )

            with tab3:
                st.dataframe(
                    st.session_state.df_scaled,
                    use_container_width=True,
                    height=300
                )
                st.caption("⚠️ Data has been Min-Max scaled to [0, 1] range")

            # Feature Selection Section
            st.markdown("---")
            st.markdown('<h2 class="neon-blue">🎯 Feature Selection</h2>', unsafe_allow_html=True)

            # Use df_encoded to ensure categorical features (now numeric) ARE selectable
            all_numeric_cols = st.session_state.df_encoded.select_dtypes(include=[np.number]).columns.tolist()

            # Select target column
            target_col = st.selectbox(
                "เลือกคอลัมน์ที่เป็นคำตอบ (Target)",
                all_numeric_cols,
                index=len(all_numeric_cols)-1 if len(all_numeric_cols) > 0 else 0,
                help="เลือกคอลัมน์เป้าหมายสำหรับการฝึก"
            )

            # Select features to use
            remaining_cols = [c for c in all_numeric_cols if c != target_col]
            selected_features = st.multiselect(
                "เลือกคอลัมน์ที่จะใช้คำนวณ (Features)",
                remaining_cols,
                default=remaining_cols,
                help="เลือก Features ที่จะใช้สำหรับการทำนาย"
            )

            # Update session state
            st.session_state.feature_cols = selected_features
            st.session_state.target_col = target_col

            # ============================================
            # Weight & Parameter Configuration Section
            # ============================================
            st.markdown("---")
            st.markdown('<h2 class="neon-blue">⚖️ Initial Parameters (W, θ, α)</h2>', unsafe_allow_html=True)

            st.markdown("""
            <div class="cyber-box">
                <p style="color: #E0E0E0;">
                    กำหนดค่าเริ่มต้น: น้ำหนัก (W), Threshold (θ) และ Learning Rate (α)<br>
                    <span style="color:#00F2FF;">สูตร: z = Σ(wᵢ·xᵢ) − θ, ŷ = 1 ถ้า z ≥ 0</span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Dynamic weight number inputs based on selected features
            if selected_features:
                n_feat = len(selected_features)
                max_cols = min(n_feat, 4)
                weight_input_cols = st.columns(max_cols)

                weight_inputs = []
                for i, feature in enumerate(selected_features):
                    with weight_input_cols[i % max_cols]:
                        w_val = st.number_input(
                            f"W{i+1} ({feature})",
                            value=0.5,
                            format="%.4f",
                            key=f"w_input_{feature}",
                            help=f"ค่าน้ำหนักเริ่มต้น w{i+1} สำหรับ {feature}"
                        )
                        weight_inputs.append(w_val)

                st.session_state.weights = weight_inputs

                # Show current initial θ and α from sidebar
                param_c1, param_c2 = st.columns(2)
                with param_c1:
                    st.info(f"**θ (Threshold) เริ่มต้น:** `{st.session_state.threshold:.4f}`  \n*(ปรับได้ที่ Sidebar)*")
                with param_c2:
                    st.info(f"**α (Learning Rate):** `{st.session_state.learning_rate:.4f}`  \n*(ปรับได้ที่ Sidebar)*")

            else:
                st.warning("⚠️ โปรดเลือก Features ก่อน")

            # Visualization Section
            st.markdown("---")
            st.markdown('<h2 class="neon-blue">📉 Visualization</h2>', unsafe_allow_html=True)

            # Create two columns for charts
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("""
                <div class="cyber-box">
                    <h3 class="neon-purple">🎯 Activation Function</h3>
                </div>
                """, unsafe_allow_html=True)

                # Generate activation curve
                x_curve, y_curve = get_activation_curve(
                    activation_internal,
                    threshold,
                    gain
                )

                # Create activation function plot
                fig_activation = go.Figure()

                # Add the activation curve
                fig_activation.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode='lines',
                    name='Activation Function',
                    line=dict(
                        color='#00F2FF',
                        width=3,
                        shape='spline'
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(0, 242, 255, 0.2)'
                ))

                # Add threshold line
                fig_activation.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="#BC13FE",
                    annotation_text="z=0",
                    annotation_position="top right"
                )

                fig_activation.update_layout(
                    title=dict(
                        text=f"{activation_type} Activation  (z = Σwᵢxᵢ − θ)",
                        font=dict(color='#00F2FF', size=14, family='Orbitron')
                    ),
                    xaxis=dict(
                        title=dict(text="z = Σ(wᵢxᵢ) − θ", font=dict(color='#E0E0E0')),
                        tickfont=dict(color='#E0E0E0'),
                        gridcolor='rgba(188, 19, 254, 0.2)',
                        zerolinecolor='#BC13FE'
                    ),
                    yaxis=dict(
                        title=dict(text="Output (ŷ)", font=dict(color='#E0E0E0')),
                        tickfont=dict(color='#E0E0E0'),
                        gridcolor='rgba(188, 19, 254, 0.2)',
                        zerolinecolor='#BC13FE',
                        range=[-0.1, 1.1]
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13, 17, 23, 0.8)',
                    font=dict(color='#E0E0E0'),
                    showlegend=False,
                    height=400,
                    margin=dict(l=50, r=50, t=60, b=50)
                )

                st.plotly_chart(fig_activation, use_container_width=True)

            with chart_col2:
                st.markdown("""
                <div class="cyber-box">
                    <h3 class="neon-emerald">📊 Weight Analysis</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create weight visualization (Radar or Bar chart)
                fig_weights = go.Figure()

                # Bar chart for weights
                feat_labels = [f"w{i+1} ({f})" for i, f in enumerate(selected_features)] if selected_features else [f"w{i+1}" for i in range(len(feature_cols))]
                w_vals = st.session_state.weights if st.session_state.weights else [0.5] * len(feature_cols)

                fig_weights.add_trace(go.Bar(
                    x=feat_labels,
                    y=w_vals,
                    marker=dict(
                        color=w_vals,
                        colorscale=[
                            [0, '#BC13FE'],
                            [0.5, '#00F2FF'],
                            [1, '#00FF9F']
                        ],
                        line=dict(color='#00F2FF', width=2)
                    ),
                    text=[f"{w:.4f}" for w in w_vals],
                    textposition='auto',
                    textfont=dict(color='#E0E0E0')
                ))

                fig_weights.update_layout(
                    title=dict(
                        text="Initial Feature Weights",
                        font=dict(color='#00FF9F', size=16, family='Orbitron')
                    ),
                    xaxis=dict(
                        title=dict(text="Features", font=dict(color='#E0E0E0')),
                        tickfont=dict(color='#E0E0E0'),
                        gridcolor='rgba(188, 19, 254, 0.2)'
                    ),
                    yaxis=dict(
                        title=dict(text="Weight Value", font=dict(color='#E0E0E0')),
                        tickfont=dict(color='#E0E0E0'),
                        gridcolor='rgba(188, 19, 254, 0.2)',
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(13, 17, 23, 0.8)',
                    font=dict(color='#E0E0E0'),
                    showlegend=False,
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )

                st.plotly_chart(fig_weights, use_container_width=True)

            # ============================================
            # Training Results Section
            # ============================================
            if st.session_state.training_completed and st.session_state.training_history:
                st.markdown("### 📈 Training Progress (Total Error per Epoch)")
                if st.session_state.epoch_errors:
                    fig_error = go.Figure()
                    fig_error.add_trace(go.Scatter(
                        x=list(range(1, len(st.session_state.epoch_errors) + 1)),
                        y=st.session_state.epoch_errors,
                        mode='lines+markers',
                        name='Total Error',
                        line=dict(color='#FF6B6B', width=3),
                        marker=dict(size=8, color='#00F2FF', symbol='circle')
                    ))
                    
                    fig_error.update_layout(
                        title=dict(
                            text="Epoch-wise Error Convergence",
                            font=dict(color='#FF6B6B', size=16, family='Orbitron')
                        ),
                        xaxis=dict(
                            title="Epoch",
                            gridcolor='rgba(188, 19, 254, 0.1)',
                            tickfont=dict(color='#E0E0E0'),
                            dtick=1
                        ),
                        yaxis=dict(
                            title="Sum of Absolute Errors (Σ|ε|)",
                            gridcolor='rgba(188, 19, 254, 0.1)',
                            tickfont=dict(color='#E0E0E0')
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(13, 17, 23, 0.8)',
                        font=dict(color='#E0E0E0'),
                        height=400,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    st.plotly_chart(fig_error, use_container_width=True)

                st.markdown("---")
                st.markdown('<h2 class="neon-blue">📊 Training Results — Step-by-Step Calculation Table</h2>', unsafe_allow_html=True)

                st.markdown("""
                <div class="cyber-box">
                  <p style="color:#00F2FF; font-size:0.95rem;">
                    <strong>สูตรที่ใช้ (ตามสไลด์หน้า 16 & 18):</strong><br>
                    z = Σ(wᵢ·xᵢ) − θ &nbsp;|&nbsp;
                    ŷ = 1 ถ้า z ≥ 0, 0 ถ้า z &lt; 0 &nbsp;|&nbsp;
                    ε = y − ŷ &nbsp;|&nbsp;
                    Δwᵢ = α·ε·xᵢ &nbsp;|&nbsp;
                    Δθ = −α·ε &nbsp;|&nbsp;
                    w_new = w_old + Δw &nbsp;|&nbsp;
                    θ_new = θ_old + Δθ
                  </p>
                  <p style="color:#BC13FE; font-size:0.85rem;">
                    ⚡ Sequential (Immediate) Update: ค่า w และ θ ถูกปรับทันทีหลังคำนวณ Error แต่ละแถว
                    แล้วนำค่าใหม่ไปใช้คำนวณแถวถัดไป
                  </p>
                </div>
                """, unsafe_allow_html=True)

                # Use features used DURING training to prevent IndexError
                feature_names = getattr(st.session_state, 'trained_feature_names', st.session_state.feature_cols)

                # Build the results DataFrame
                results_data = []
                for log_entry in st.session_state.training_history:
                    # Specific column ordering as per Final requirements:
                    # n, Epoch, x1, x2, y (Desired), y_hat (Predicted), Err (ε), W1_New, Δw1, W2_New, Δw2, θ_New, Δθ
                    row = {
                        'n': log_entry['n'],
                        'Epoch': log_entry['epoch'],
                    }

                    # Input features (x1, x2, ...)
                    for i, feat_name in enumerate(feature_names):
                        row[f'x{i+1}'] = round(log_entry['inputs'][i], 4)

                    # Desired output y
                    row['y (Desired)'] = float(log_entry['desired'])

                    # Predicted y_hat
                    row['y_hat (Predicted)'] = float(log_entry['predicted'])

                    # Error (ε)
                    row['Err (ε)'] = float(log_entry['error'])

                    # New Weights and Deltas (W#_New, Δw#)
                    for i, feat_name in enumerate(feature_names):
                        row[f'W{i+1}_New'] = round(log_entry['weights_after'][i], 4)
                        row[f'Δw{i+1}'] = round(log_entry['weight_deltas'][i], 4)

                    # Threshold (θ_New, Δθ)
                    row['θ_New'] = round(log_entry['theta_after'], 4)
                    row['Δθ'] = round(log_entry['theta_delta'], 4)

                    results_data.append(row)

                results_df = pd.DataFrame(results_data)
                # Set 'n' as index to replace default integer index
                results_df = results_df.set_index('n')

                # Color-code error rows: highlight rows with error ≠ 0
                def highlight_error(row):
                    if abs(row['Err (ε)']) > 1e-6:
                        return ['background-color: rgba(255, 100, 100, 0.15)'] * len(row)
                    return [''] * len(row)

                st.markdown("### 📋 Detailed Training Log (4 decimal places)")
                styled_df = results_df.style.apply(highlight_error, axis=1).format(
                    "{:.4f}", subset=results_df.select_dtypes(include=[np.number]).columns
                )
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=520
                )

                # Summary metrics
                st.markdown("### 📈 Training Summary")
                col1, col2, col3, col4 = st.columns(4)

                total_steps = len(results_df)
                epochs_done = int(results_df['Epoch'].max()) if not results_df.empty else 0
                total_errors = int(results_df['Err (ε)'].abs().sum())
                final_theta = st.session_state.final_threshold

                with col1:
                    st.metric("Total Steps (n)", total_steps)
                with col2:
                    st.metric("Epochs Completed", epochs_done)
                with col3:
                    st.metric("Total Errors (Σ|ε|)", total_errors)
                with col4:
                    st.metric("Final θ", f"{final_theta:.4f}")

                # Final weights summary
                st.markdown("### ⚖️ Final Weights after Training")
                fw = st.session_state.final_weights
                if fw is not None:
                    fw_cols = st.columns(len(feature_names) + 1)
                    for i, fname in enumerate(feature_names):
                        with fw_cols[i]:
                            st.metric(f"w{i+1} ({fname})", f"{fw[i]:.4f}")
                    with fw_cols[len(feature_names)]:
                        st.metric("θ_final", f"{final_theta:.4f}")

                # Download results as CSV (Placed below training results/summary)
                st.markdown("### 📥 Export Training Data")
                csv_buffer = results_df.to_csv(index=True)
                st.download_button(
                    label="⬇️ Download Training Log (CSV)",
                    data=csv_buffer,
                    file_name=f"perceptron_training_log_{uploaded_file.name}.csv",
                    mime="text/csv",
                    key="download_results",
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("""
            <div class="neon-box" style="margin-top: 30px;">
                <h2 class="neon-emerald">🎮 Manual Prediction Box</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="cyber-box">
                <p style="color: #E0E0E0;">
                    Enter values for each feature to get a prediction from the perceptron.
                    The values will be automatically scaled using the same Min-Max scaling.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Input fields for manual prediction
            pred_col1, pred_col2, pred_col3 = st.columns([1, 2, 1])

            with pred_col1:
                st.markdown("### 📥 Input Values")

                input_values = []
                input_cols = st.columns(2)

                for i, col in enumerate(feature_cols):
                    with input_cols[i % 2]:
                        if col in st.session_state.mappings:
                            # Categorical feature: Use Selectbox
                            mapping = st.session_state.mappings[col]
                            options = list(mapping.values())
                            choice = st.selectbox(
                                f"{col}",
                                options,
                                key=f"pred_{i}"
                            )
                            # Convert choice back to numeric code
                            val = [k for k, v in mapping.items() if v == choice][0]
                        else:
                            # Numeric feature: Use Number Input
                            orig_min = float(st.session_state.df_encoded[col].min())
                            orig_max = float(st.session_state.df_encoded[col].max())
                            val = st.number_input(
                                f"{col}",
                                min_value=orig_min,
                                max_value=orig_max,
                                value=(orig_min + orig_max) / 2.0,
                                step=0.1,
                                format="%.4f",
                                key=f"pred_{i}"
                            )
                        input_values.append(val)

            with pred_col2:
                # Prediction Button
                st.markdown("<br>" * 3, unsafe_allow_html=True)
                if st.button("🚀 CALCULATE PREDICTION", use_container_width=True):
                    st.session_state.calculation_done = True

                if st.session_state.calculation_done:
                    # Perform prediction
                    # Scale input values
                    input_scaled = []
                    for i, col in enumerate(feature_cols):
                        orig_min = float(st.session_state.df_encoded[col].min())
                        orig_max = float(st.session_state.df_encoded[col].max())
                        if orig_max - orig_min != 0:
                            scaled_val = (input_values[i] - orig_min) / (orig_max - orig_min)
                        else:
                            scaled_val = 0
                        input_scaled.append(scaled_val)

                    # Use final weights if training is done, else initial weights
                    w_for_pred = st.session_state.final_weights if (
                        st.session_state.training_completed and st.session_state.final_weights is not None
                    ) else np.array(st.session_state.weights)

                    theta_for_pred = st.session_state.final_threshold if (
                        st.session_state.training_completed and st.session_state.final_threshold is not None
                    ) else st.session_state.threshold

                    # Calculate prediction
                    prediction, net_input, raw_output = perceptron_predict(
                        np.array(input_scaled),
                        np.array(w_for_pred),
                        activation_internal,
                        theta_for_pred,
                        gain
                    )

                    st.session_state.last_prediction = prediction[0] if hasattr(prediction, '__len__') else prediction
                    st.session_state.last_net_input = net_input if not hasattr(net_input, '__len__') else float(net_input)
                    st.session_state.last_raw_output = raw_output if not hasattr(raw_output, '__len__') else float(raw_output)
                    st.session_state.last_input_scaled = input_scaled

            with pred_col3:
                if st.session_state.calculation_done:
                    pred = st.session_state.last_prediction
                    raw = st.session_state.last_raw_output
                    net = st.session_state.last_net_input

                    st.markdown("### 📤 Results")

                    # Display prediction result
                    result_color = "#00FF9F" if pred == 1 else "#FF6B6B"
                    
                    # Try to get the original label name if it was encoded
                    map_info = st.session_state.mappings.get(st.session_state.target_col, {})
                    label_name = map_info.get(int(pred), f"CLASS {int(pred)}")
                    result_text = f"{label_name}"

                    st.markdown(f"""
                    <div class="cyber-box" style="text-align: center; border-color: {result_color};">
                        <h2 style="color: {result_color}; font-size: 3rem; margin: 0;">
                            {result_text}
                        </h2>
                        <p style="color: #E0E0E0;">
                            Net Input (z): <span style="color: #00F2FF;">{net:.4f}</span><br>
                            Raw Output: <span style="color: #BC13FE;">{raw:.4f}</span>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar for confidence
                    st.markdown("### 📊 Confidence Level")
                    confidence = abs(raw) if activation_internal == 'sigmoid' else (1.0 if net >= 0 else 0.0)
                    progress_val = min(max(float(confidence), 0), 1)

                    st.progress(progress_val)
                    st.caption(f"Signal Strength: {progress_val*100:.1f}%")

else:
    # Welcome Screen (when no file uploaded)
    st.markdown("""
    <div class="neon-box" style="text-align: center; padding: 50px;">
        <h2 class="neon-blue">🚀 GET STARTED</h2>
        <p style="color: #E0E0E0; font-size: 1.3rem; margin: 30px 0;">
            Upload a dataset (CSV or ARFF) from the sidebar to begin your Perceptron analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Try to show Lottie animation
    try:
        from streamlit_lottie import st_lottie
        lottie_data = load_lottieurl(LOTTIE_DATA)
        if lottie_data:
            st_lottie(lottie_data, height=300, key="welcome_anim")
    except:
        st.markdown("📊")

    # Features showcase
    st.markdown("""
    <div class="cyber-box">
        <h3 class="neon-purple">✨ Features</h3>
        <ul style="color: #E0E0E0; font-size: 1.1rem; line-height: 2;">
            <li>📂 Support for CSV and ARFF file formats</li>
            <li>🧹 Automatic data cleaning (missing values)</li>
            <li>📊 Min-Max data scaling</li>
            <li>🎯 Interactive activation function visualization</li>
            <li>⚖️ Real-time weight adjustment (st.number_input — free decimal entry)</li>
            <li>📋 Step-by-step training table with z, ŷ, ε, Δw, Δθ columns</li>
            <li>🔥 Sequential (immediate) weight update — slide 16 &amp; 18 method</li>
            <li>🎮 Manual prediction with live feedback</li>
            <li>📈 Confidence level indicator</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #BC13FE; font-family: 'Orbitron';">
    <p>🧠 NeuroCore Dashboard | Powered by Streamlit &amp; Plotly</p>
    <p style="font-size: 0.8rem; color: #E0E0E0;">
        Perceptron Neural Network Simulator v2.0 — Sequential Update Mode
    </p>
</div>
""", unsafe_allow_html=True)
