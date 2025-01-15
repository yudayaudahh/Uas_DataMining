import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Setup Layout
st.set_page_config(
    page_title="📊 Multiple Linear Regression App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS untuk meningkatkan estetika
st.markdown("""
    <style>
    /* Gaya umum */
    body {
        background-color: #F5F5F5;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #34495E;
    }
    .stButton>button {
        background: linear-gradient(to right, #1ABC9C, #16A085);
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #16A085, #1ABC9C);
    }
    .stSidebar {
        background-color: #ECF0F1;
        padding: 10px;
        border-radius: 10px;
    }
    .stDataFrame {
        background-color: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #D5D8DC;
        padding: 10px;
    }
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        gap: 15px;
    }
    .metric {
        background: #EAF2F8;
        border: 1px solid #D6EAF8;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-size: 18px;
        color: #34495E;
    }
    </style>
""", unsafe_allow_html=True)

# Judul Aplikasi
st.title("📊 Multiple Linear Regression App")
st.subheader("Effortlessly build and analyze predictive models with style!")

# Membaca dataset langsung dari file yang sudah ada
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Path dataset
file_path = r"D:\DATA_MINING\Regression.csv"
try:
    data = load_data(file_path)
    st.markdown("## 🗂 Dataset Overview")
    st.dataframe(data.style.highlight_max(axis=0), use_container_width=True)
except FileNotFoundError:
    st.error(f"❌ File not found at: {file_path}. Please check the file path and try again.")
    st.stop()

# Sidebar
st.sidebar.header("🔧 Configure Model")
st.sidebar.markdown("Set up your features and target for analysis.")
features = st.sidebar.multiselect("🔹 Select Features (X)", data.columns)
target = st.sidebar.selectbox("🔸 Select Target (Y)", data.columns)

if features and target:
    X = data[features]
    
    y = data[target]

    # Split Data
    st.sidebar.markdown("### ⚙️ Data Splitting")
    test_size = st.sidebar.slider("📐 Test Size (Proportion)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("🎲 Random State (Optional)", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi Model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Tampilkan Hasil
    st.markdown("## 📊 Model Results")
    st.markdown("""
    <div class="metric-container">
        <div class="metric">📊 <b>Mean Squared Error:</b><br>{:.2f}</div>
        <div class="metric">📈 <b>R-Squared:</b><br>{:.2f}</div>
        <div class="metric">⚙️ <b>Intercept:</b><br>{:.2f}</div>
    </div>
    """.format(mse, r2, model.intercept_), unsafe_allow_html=True)

    # Koefisien
    st.markdown("### 📉 Coefficients")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    st.dataframe(coef_df.style.background_gradient(cmap="cool"), use_container_width=True)

    # Grafik: Actual vs Predicted
    st.markdown("## 📈 Actual vs Predicted")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name="Predictions", marker=dict(color="blue")))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name="Ideal Line", line=dict(color='red', dash='dash')))
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        legend_title="Legend",
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribusi Error
    st.markdown("## 📊 Residuals Distribution")
    residuals = y_test - y_pred
    fig_residuals = px.histogram(
        residuals, 
        title="Residuals Distribution", 
        nbins=20, 
        template="plotly_white",
        color_discrete_sequence=["#3498DB"]
    )
    fig_residuals.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
    st.plotly_chart(fig_residuals, use_container_width=True)

else:
    st.sidebar.warning("⚠️ Please select features and target in the sidebar.")
    st.markdown("### 🚀 Start Here")
    st.info("Select features and target from the sidebar to proceed!")
