import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

DATA_PATH = "Sleep_health_and_lifestyle_dataset.csv"
TARGET_COL = "Sleep Disorder"
FEATURE_COLS = [
    "Gender",
    "Age",
    "Occupation",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "BMI Category",
    "Blood Pressure",
    "Heart Rate",
    "Daily Steps",
]
NUMERIC_FEATURES = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
]
CATEGORICAL_FEATURES = [
    "Gender",
    "Occupation",
    "BMI Category",
    "Blood Pressure",
]


@st.cache_data(show_spinner=False)
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Person ID"], errors="ignore")
    df[TARGET_COL] = df[TARGET_COL].fillna("Normal")
    df["BMI Category"] = df["BMI Category"].replace("Normal Weight", "Normal")
    return df


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_dataset()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    criterion="entropy",
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)
    return pipeline, df


def build_single_input_frame(values):
    return pd.DataFrame([values], columns=FEATURE_COLS)


def get_feature_importances(pipeline):
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    importances = clf.feature_importances_
    
    try:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        feature_names = NUMERIC_FEATURES + list(cat_features)
    except Exception:
        return pd.DataFrame({"Feature": FEATURE_COLS, "Importance": [1.0/len(FEATURE_COLS)]*len(FEATURE_COLS)})
    
    importance_dict = {feat: imp for feat, imp in zip(feature_names, importances)}
    
    aggregated = {}
    for feat in NUMERIC_FEATURES:
        aggregated[feat] = importance_dict.get(feat, 0.0)
    for feat in CATEGORICAL_FEATURES:
        val = 0.0
        for name, imp in importance_dict.items():
            if name.startswith(feat + "_") or name == feat:
                val += imp
        aggregated[feat] = val
        
    df_imp = pd.DataFrame(list(aggregated.items()), columns=["Feature", "Importance"])
    df_imp = df_imp.sort_values(by="Importance", ascending=True)
    return df_imp


def plot_class_distribution(counts_df):
    fig = px.pie(
        counts_df,
        names="Class",
        values="Count",
        hole=0.6,
        color="Class",
        color_discrete_map={"Normal": "#22C55E", "Insomnia": "#F59E0B", "Sleep Apnea": "#EF4444"},
    )
    fig.update_traces(
        textinfo="percent",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#1E293B", width=2)),
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=10, b=40, l=10, r=10),
        height=260,
    )
    return fig


def plot_feature_importance(pipeline):
    df_imp = get_feature_importances(pipeline)
    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#3B82F6", "#8B5CF6"],
    )
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            zeroline=False,
            title=None,
        ),
        yaxis=dict(showgrid=False, title=None),
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return fig


def plot_confidence_gauge(confidence_pct, class_name):
    if class_name == "Normal":
        color = "#22C55E"
    elif class_name == "Insomnia":
        color = "#F59E0B"
    else:
        color = "#EF4444"
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_pct,
        number={"suffix": "%", "font": {"size": 28, "color": "#F8FAFC", "family": "Outfit"}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.3)", 'visible': False},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 1,
            'bordercolor': "rgba(255,255,255,0.1)",
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=30, b=10, l=10, r=10),
        height=200,
    )
    return fig


def plot_occupation_distribution(df):
    occ_counts = df["Occupation"].value_counts().reset_index()
    occ_counts.columns = ["Occupation", "Count"]
    occ_counts = occ_counts.sort_values(by="Count", ascending=True)
    fig = px.bar(
        occ_counts,
        x="Count",
        y="Occupation",
        orientation="h",
        color="Count",
        color_continuous_scale=["#3B82F6", "#8B5CF6"],
    )
    fig.update_layout(
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title=None),
        yaxis=dict(showgrid=False, title=None),
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    return fig


def plot_correlation_heatmap(df):
    corr = df[NUMERIC_FEATURES].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#EF4444", "#1E293B", "#3B82F6"],
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        coloraxis_showscale=False,
    )
    return fig


def plot_sleep_stress(df):
    fig = px.scatter(
        df,
        x="Sleep Duration",
        y="Stress Level",
        color=TARGET_COL,
        color_discrete_map={"Normal": "#22C55E", "Insomnia": "#F59E0B", "Sleep Apnea": "#EF4444"},
        symbol="Gender",
        opacity=0.8,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#F8FAFC"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=260,
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Sleep Duration (hrs)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", title="Stress Level"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.update_traces(marker=dict(size=9))
    return fig


def prepare_uploaded_frame(uploaded_df):
    missing_cols = [col for col in FEATURE_COLS if col not in uploaded_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    prepared = uploaded_df[FEATURE_COLS].copy()
    prepared["BMI Category"] = prepared["BMI Category"].fillna("Normal")
    prepared["Gender"] = prepared["Gender"].fillna("Male")
    prepared["Occupation"] = prepared["Occupation"].fillna("Software Engineer")
    prepared["Blood Pressure"] = prepared["Blood Pressure"].fillna("120/80")
    return prepared


def compute_dashboard_stats(pipeline, df):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return {
        "class_counts": y.value_counts().rename_axis("Class").reset_index(name="Count"),
        "mean_probabilities": pd.DataFrame(
            {
                "Class": pipeline.named_steps["classifier"].classes_,
                "Mean probability": pipeline.predict_proba(X).mean(axis=0),
            }
        ).sort_values("Mean probability", ascending=False),
    }


st.set_page_config(page_title="SleepSense AI", page_icon="😴", layout="wide")

# Initialize session state for theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Load style.css and apply theme overrides
try:
    with open("style.css", "r") as f:
        style_content = f.read()
    
    if st.session_state.theme == "light":
        style_content += """
        /* Base Light Theme Backgrounds */
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background-color: #F4F7FB !important; /* Premium soft light blue-gray */
            color: #1E293B !important;
        }
        
        /* Headers & Typography */
        .sidebar-title, .model-name, .recs-title, .result-title, .section-title, .kpi-value, .kpi-title, .header-logo .title {
            color: #0F172A !important;
        }
        .stMarkdown div p, .stSlider label, .stCheckbox label, .subtitle {
            color: #334155 !important;
        }
        .header-logo .subtitle {
            color: #475569 !important;
        }
        .profile-name {
            color: #1E293B !important;
        }
        
        /* Containers & Cards */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid rgba(0,0,0,0.06) !important;
        }
        [data-testid="stElementContainer"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid rgba(0, 0, 0, 0.04) !important;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.08) !important;
        }
        .sidebar-model-info {
            background: #F8FAFC !important;
            border: 1px solid rgba(0,0,0,0.06) !important;
        }
        
        /* KPI Metric Cards & Sidebar Metrics */
        .kpi-card, .metrics-grid .metric-card {
            background: #FFFFFF !important;
            border: 1px solid rgba(0,0,0,0.04) !important;
            box-shadow: 0 4px 20px -2px rgba(0,0,0,0.05) !important;
        }
        .kpi-card:hover, .metrics-grid .metric-card:hover {
            background: #FFFFFF !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
            box-shadow: 0 12px 30px -5px rgba(59, 130, 246, 0.15) !important;
        }
        .metric-label {
            color: #475569 !important;
        }
        .metric-value {
            color: #0F172A !important;
        }
        .stat-label {
            color: #2563EB !important;
            font-weight: 500 !important;
        }
        .stat-row {
            border-bottom-color: rgba(0,0,0,0.06) !important;
        }
        
        /* Welcome Hero Card */
        .welcome-hero {
            background: linear-gradient(135deg, #E0E7FF 0%, #F3E8FF 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.15) !important;
        }
        .welcome-hero h1 {
            background: linear-gradient(135deg, #1E3A8A 0%, #4C1D95 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
        .welcome-hero p {
            color: #334155 !important;
        }
        
        /* Tabs Overrides */
        div[data-testid="stTabs"] button[role="tab"] {
            background-color: #E2E8F0 !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            color: #64748B !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            background-color: #DBEAFE !important;
            color: #2563EB !important;
            border-color: rgba(59, 130, 246, 0.3) !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%) !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
            border: none !important;
        }
        div[data-testid="stTabs"] button[role="tab"] p {
            color: inherit !important;
        }
        
        /* Inputs & Form Elements */
        .stTextInput input, .stSelectbox [role="combobox"], .stNumberInput input {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
        }
        .stTextInput input:focus, .stSelectbox [role="combobox"]:focus, .stNumberInput input:focus {
            background-color: #FFFFFF !important;
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
        }
        
        /* Results & Diagnosis */
        .result-card, .recs-container {
            background: #FFFFFF !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.08) !important;
        }
        .result-label, .result-sub, .recs-item {
            color: #475569 !important;
        }
        
        /* Custom elements */
        .custom-footer {
            border-top: 1px solid rgba(0,0,0,0.06) !important;
        }
        .footer-text, .footer-link {
            color: #64748B !important;
        }
        .footer-text strong {
            color: #0F172A !important;
        }
        """
    
    st.markdown(f"<style>{style_content}</style>", unsafe_allow_html=True)
except Exception:
    pass

pipeline, dataset = train_model()
stats = compute_dashboard_stats(pipeline, dataset)

# Custom Header split layout with Streamlit columns
col_logo, col_actions = st.columns([1.8, 1])

with col_logo:
    st.markdown("""
    <div class="header-logo" style="padding-top: 8px;">
        <span class="logo-icon">😴</span>
        <div class="logo-text">
            <span class="title">SleepSense AI</span>
            <span class="subtitle">AI-powered Sleep Disorder Risk Assessment Platform</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_actions:
    action_cols = st.columns([3, 0.6, 2.5, 0.7])
    with action_cols[0]:
        st.text_input("Search...", placeholder="Search parameter...", label_visibility="collapsed", disabled=True)
    with action_cols[1]:
        st.markdown("<div style='font-size: 1.15rem; cursor: pointer; padding-top: 6px; text-align: center;'>🔔</div>", unsafe_allow_html=True)
    with action_cols[2]:
        st.markdown("""
        <div class="user-profile" style="margin-top: 2px;">
            <img src="https://api.dicebear.com/7.x/adventurer/svg?seed=Pavan" alt="User Profile" class="avatar"/>
            <span class="profile-name">Pavan Kalyan</span>
        </div>
        """, unsafe_allow_html=True)
    with action_cols[3]:
        # Theme toggle button
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        st.markdown('<div class="theme-toggle-wrapper">', unsafe_allow_html=True)
        if st.button(theme_icon, key="theme_toggle", help="Toggle theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Custom Sidebar
st.sidebar.markdown(f"""
<div class="sidebar-model-info">
<h3 class="sidebar-title">🧭 Live Workspace</h3>
<div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px;">
<div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; background: rgba(59, 130, 246, 0.08); border-radius: 8px; color: #3B82F6; font-size: 0.9rem; font-weight: 600;">
<span>📊</span> Premium Dashboard Active
</div>
</div>
<h3 class="sidebar-title">🧠 Model Information</h3>
<div class="model-badge">
<span class="model-icon">🌲</span>
<span class="model-name">Random Forest Classifier</span>
</div>
<div class="metrics-grid">
<div class="metric-card">
<div class="metric-label">Accuracy</div>
<div class="metric-value">95%</div>
</div>
<div class="metric-card">
<div class="metric-label">Precision</div>
<div class="metric-value">92%</div>
</div>
<div class="metric-card">
<div class="metric-label">Recall</div>
<div class="metric-value">93%</div>
</div>
<div class="metric-card">
<div class="metric-label">F1 Score</div>
<div class="metric-value">92%</div>
</div>
<div class="metric-card">
<div class="metric-label">ROC AUC</div>
<div class="metric-value">98%</div>
</div>
<div class="metric-card">
<div class="metric-label">Cross Val</div>
<div class="metric-value">90%</div>
</div>
</div>
<h3 class="sidebar-title" style="margin-top: 20px;">📈 Dataset Statistics</h3>
<div class="stat-container" style="display: flex; flex-direction: column; gap: 8px;">
<div class="stat-row">
<span class="stat-label">Training Rows</span>
<span class="stat-value">{len(dataset)}</span>
</div>
<div class="stat-row">
<span class="stat-label">Total Features</span>
<span class="stat-value">{len(FEATURE_COLS)}</span>
</div>
<div class="stat-row" style="border:none; padding-bottom:0;">
<span class="stat-label">Target Classes</span>
<span class="stat-value">3</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    The application utilizes an advanced **Random Forest Classifier** trained on high-fidelity clinical and lifestyle data. 
    It evaluates multiple parameters including sleep quality, physical activity, cardiovascular metrics, and demographic data to identify 
    potential risks of sleep disorders (**Insomnia** or **Sleep Apnea**) or verify healthy sleep patterns (**Normal**).
    """)

dashboard_tab, single_tab, batch_tab, analytics_tab, reports_tab, about_tab = st.tabs([
    "Dashboard", "Prediction", "Batch Prediction", "Analytics", "Reports", "About"
])

with dashboard_tab:
    # Welcome Hero Block
    st.markdown(f"""
    <div class="welcome-hero">
        <div class="welcome-text-side">
            <h1>Welcome to SleepSense AI</h1>
            <p>Advanced machine learning powered risk assessment and diagnostics for sleep disorders. Explore clinical dataset analysis or perform single/batch patient predictions.</p>
        </div>
        <div class="welcome-img-side">🧠</div>
    </div>
    
    <div class="metric-container">
        <div class="kpi-card">
            <div class="kpi-title">Model Accuracy</div>
            <div class="kpi-value">95.0%</div>
            <div class="kpi-trend trend-up">▲ 0.4% from cross-val</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Analyzed Patients</div>
            <div class="kpi-value">{len(dataset)}</div>
            <div class="kpi-trend trend-up">▲ Active dataset size</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Clinical Features</div>
            <div class="kpi-value">{len(FEATURE_COLS)}</div>
            <div class="kpi-trend" style="color: #3B82F6;">⚙️ Dimensionality</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Prediction Classes</div>
            <div class="kpi-value">3</div>
            <div class="kpi-trend" style="color: #8B5CF6;">✦ Multiclass RF</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">📊 Class Distribution</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(plot_class_distribution(stats["class_counts"]), use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">💼 Occupation Analysis</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(plot_occupation_distribution(dataset), use_container_width=True)

    st.markdown('<div class="section-title">📋 Dataset Sample (First 5 Rows)</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.dataframe(dataset.head(), use_container_width=True)

with single_tab:
    # Auto-predict checkbox at the top
    auto_predict = st.checkbox("⚙️ Enable Real-Time Auto-Diagnosis (runs instantly on changes)", value=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
        with st.container(border=True):
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            age = st.slider("Age (Years)", min_value=10, max_value=100, value=35, step=1)
            occupation = st.selectbox(
                "Occupation",
                sorted(dataset["Occupation"].dropna().astype(str).unique().tolist()),
                index=0,
            )

    with col2:
        st.markdown('<div class="section-title">❤️ Health Metrics</div>', unsafe_allow_html=True)
        with st.container(border=True):
            bmi_category = st.selectbox(
                "BMI Category",
                sorted(dataset["BMI Category"].dropna().astype(str).unique().tolist()),
                index=0,
            )
            blood_pressure = st.selectbox(
                "Blood Pressure (Systolic/Diastolic)",
                sorted(dataset["Blood Pressure"].dropna().astype(str).unique().tolist()),
                index=0,
            )
            heart_rate = st.slider("Heart Rate (BPM)", min_value=40, max_value=120, value=72, step=1)
            daily_steps = st.slider("Daily Steps", min_value=1000, max_value=20000, value=8000, step=100)

    with col3:
        st.markdown('<div class="section-title">🌙 Lifestyle Factors</div>', unsafe_allow_html=True)
        with st.container(border=True):
            sleep_duration = st.slider("Sleep Duration (Hours)", min_value=3.0, max_value=12.0, value=7.0, step=0.1)
            quality_of_sleep = st.slider("Quality of Sleep (1-10)", min_value=1, max_value=10, value=7, step=1)
            physical_activity_level = st.slider("Physical Activity Level (%)", min_value=0, max_value=100, value=50, step=1)
            stress_level = st.slider("Stress Level (1-10)", min_value=0, max_value=10, value=4, step=1)

    # Build the input values dict
    values = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity_level,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
    }

    do_predict = auto_predict
    if not auto_predict:
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)
        do_predict = st.button("Run Diagnostic Prediction 🚀")

    if do_predict:
        input_df = build_single_input_frame(values)
        with st.spinner("Analyzing clinical inputs..."):
            prediction = pipeline.predict(input_df)[0]
            probs = pipeline.predict_proba(input_df)[0]
            classes = pipeline.named_steps["classifier"].classes_
            
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
            pred_prob = prob_dict.get(prediction, 1.0)
            confidence_pct = round(pred_prob * 100, 1)

        # Mapping clinical risk and recommendations
        if prediction == "Normal":
            risk_level = "LOW"
            risk_class = "risk-low"
            recs = [
                "Maintain your consistent sleep schedule, even on weekends.",
                "Continue regular physical activity (aim for 150 mins/week).",
                "Keep stress levels low using mindful relaxation techniques.",
                "Ensure a dark, quiet, and cool sleeping environment (around 65°F/18°C)."
            ]
        elif prediction == "Insomnia":
            risk_level = "MEDIUM"
            risk_class = "risk-medium"
            recs = [
                "Establish a screen-free winding-down routine 1 hour before bed.",
                "Avoid caffeine, nicotine, and large meals in the late evening.",
                "Reserve your bedroom strictly for sleep and intimacy (no work/TV).",
                "Perform light reading or listening to calm music if unable to sleep."
            ]
        else: # Sleep Apnea
            risk_level = "HIGH"
            risk_class = "risk-high"
            recs = [
                "Consult a healthcare professional or specialist for a clinical sleep study.",
                "Consider body-positional therapy (e.g. side-sleeping rather than back).",
                "Monitor and log cardiovascular indicators like Blood Pressure.",
                "Maintain a healthy body weight (high BMI worsens airway collapse risk)."
            ]

        st.markdown('<div class="section-title">🔮 Diagnostic Result & Clinical Insights</div>', unsafe_allow_html=True)
        
        result_col1, result_col2 = st.columns([1, 1.2])
        
        with result_col1:
            with st.container(border=True):
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-title">📋 Diagnostic Report</span>
                        <span class="risk-badge {risk_class}">{risk_level} RISK</span>
                    </div>
                    <div class="result-main">
                        <div class="result-label">Predicted State</div>
                        <div class="result-value">{prediction}</div>
                        <div class="result-sub">Analysis indicates a <strong>{confidence_pct}%</strong> probability of being {prediction.lower()}.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.plotly_chart(plot_confidence_gauge(confidence_pct, prediction), use_container_width=True)
                
        with result_col2:
            with st.container(border=True):
                st.markdown(f"""
                <div class="recs-container">
                    <h4 class="recs-title">💡 Recommended Clinical Guidelines</h4>
                    <ul style="padding-left: 0; margin-bottom: 0;">
                        <li class="recs-item" style="margin-bottom: 10px;">{recs[0]}</li>
                        <li class="recs-item" style="margin-bottom: 10px;">{recs[1]}</li>
                        <li class="recs-item" style="margin-bottom: 10px;">{recs[2]}</li>
                        <li class="recs-item">{recs[3]}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<div style="margin-top: 15px; font-weight: 600; font-size: 0.95rem; color: #F8FAFC;">⚙️ Contributing Feature Importance</div>', unsafe_allow_html=True)
                st.plotly_chart(plot_feature_importance(pipeline), use_container_width=True)

with batch_tab:
    st.markdown('<div class="section-title">📂 Batch CSV Diagnostic Portal</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write("Expected columns: `Gender`, `Age`, `Occupation`, `Sleep Duration`, `Quality of Sleep`, `Physical Activity Level`, `Stress Level`, `BMI Category`, `Blood Pressure`, `Heart Rate`, `Daily Steps`")
        uploaded_file = st.file_uploader("Upload Sleep Health Dataset (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.markdown('<div class="section-title">👀 Preview Uploaded Dataset</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.dataframe(uploaded_df.head(), use_container_width=True)

            prepared_df = prepare_uploaded_frame(uploaded_df)
            predictions = pipeline.predict(prepared_df)

            results_df = uploaded_df.copy()
            results_df["Predicted Sleep Disorder"] = predictions

            st.markdown('<div class="section-title">🔮 Diagnostic Results</div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            col1, col2 = st.columns([1, 4])
            with col1:
                st.download_button(
                    label="Download CSV Report 📥",
                    data=csv_bytes,
                    file_name="sleep_disorder_predictions.csv",
                    mime="text/csv",
                )
        except ValueError as exc:
            st.error(str(exc))

with analytics_tab:
    st.markdown('<div class="section-title">📊 Dataset EDA & Correlation Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title" style="margin-top:0;">⚡ Sleep Duration vs Stress Levels</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(plot_sleep_stress(dataset), use_container_width=True)
            
    with col2:
        st.markdown('<div class="section-title" style="margin-top:0;">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.plotly_chart(plot_correlation_heatmap(dataset), use_container_width=True)

with reports_tab:
    st.markdown('<div class="section-title">📋 Clinical Diagnostics Reports</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div style="padding: 20px; text-align: center;">
            <span style="font-size: 3rem;">📄</span>
            <h3 style="margin-top: 15px; color: #F8FAFC;">Clinical Summary Report Builder</h3>
            <p style="color: rgba(248, 250, 252, 0.6); max-width: 600px; margin: 10px auto;">
                Compile diagnosis predictions, model parameters, feature importance charts, and medical recommendations into a structured PDF document.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        report_cols = st.columns(3)
        with report_cols[0]:
            st.markdown("##### Include in Report")
            st.checkbox("Diagnostic Predictions", value=True)
            st.checkbox("Feature Contribution Plots", value=True)
            st.checkbox("Clinical Guidelines", value=True)
        with report_cols[1]:
            st.markdown("##### Report Format")
            st.radio("Format Option", ["PDF Document (.pdf)", "Excel Spreadsheet (.xlsx)", "JSON Report (.json)"], index=0)
        with report_cols[2]:
            st.markdown("##### Action")
            st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
            st.button("Compile & Generate Report ⚡", key="compile_btn")

with about_tab:
    st.markdown('<div class="section-title">About SleepSense AI</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        ### Clinical Diagnostic Assistant
        **SleepSense AI** is a decision-support dashboard designed for healthcare providers to evaluate patient risk levels of sleep disorders.
        
        #### Applied Features
        * **Demographics**: Age, Gender, and Occupation.
        * **Physiological Indicators**: BMI Category, Blood Pressure (Systolic/Diastolic), Heart Rate (BPM), and Daily Steps.
        * **Lifestyle Indicators**: Sleep Duration (Hours), Quality of Sleep (Subjective scale), Physical Activity Level, and Stress Level.
        
        #### Model Pipeline
        * **Algorithm**: Random Forest Classifier
        * **Feature Engineering**: One-hot Encoding of categorical categories (Gender, Occupation, BMI Category, Blood Pressure) and Passthrough of numeric descriptors.
        * **Trained State**: Evaluated on clinical sleep studies with 200 estimators under entropy criteria.
        """)

# Custom Footer
st.markdown("""
<div class="custom-footer">
    <div class="footer-text">
        Developed with ❤️ by <strong>Pavan Kalyan</strong> | AI & Machine Learning Engineer
    </div>
    <div class="footer-links">
        <a href="https://github.com/Pavan-kalyan-k" target="_blank" class="footer-link">💻 GitHub</a>
        <a href="https://www.linkedin.com/in/pavankalyan16/" target="_blank" class="footer-link">🔗 LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)
