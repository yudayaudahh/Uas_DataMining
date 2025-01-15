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

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    body { background-color: #F5F5F5; font-family: 'Arial', sans-serif; }
    h1, h2, h3, h4 { color: #34495E; }
    .stButton>button { background: linear-gradient(to right, #1ABC9C, #16A085); color: white; border: none; border-radius: 8px; font-size: 16px; padding: 10px 20px; }
    .stButton>button:hover { background: linear-gradient(to right, #16A085, #1ABC9C); }
    .stSidebar { background-color: #ECF0F1; padding: 10px; border-radius: 10px; }
    .metric-container { display: flex; flex-wrap: wrap; justify-content: space-around; gap: 15px; }
    .metric { background: #EAF2F8; border: 1px solid #D6EAF8; border-radius: 8px; padding: 15px; text-align: center; font-size: 18px; color: #34495E; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Multiple Linear Regression App")
st.subheader("Effortlessly build and analyze predictive models with style!")

# Load dataset
@st.cache_data
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

file_path = r"Regression.csv"
try:
    data = load_data(file_path)
    st.markdown("## 🗂 Dataset Overview")
    st.dataframe(data, use_container_width=True)
except FileNotFoundError:
    st.error(f"❌ File not found at: {file_path}. Please check the file path and try again.")
    st.stop()

# Sidebar Configuration
st.sidebar.header("🔧 Configure Model")
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

features = st.sidebar.multiselect("🔹 Select Features (X)", numeric_columns)
target = st.sidebar.selectbox("🔸 Select Target (Y)", numeric_columns)

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

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Results Display
    st.markdown("## 📊 Model Results")
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric">📊 <b>Mean Squared Error:</b><br>{mse:.2f}</div>
        <div class="metric">📈 <b>R-Squared:</b><br>{r2:.2f}</div>
        <div class="metric">⚙️ <b>Intercept:</b><br>{model.intercept_:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Coefficients
    st.markdown("### 📉 Coefficients")
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    st.dataframe(coef_df, use_container_width=True)

    # Actual vs Predicted Plot
    st.markdown("## 📈 Actual vs Predicted")
    fig = go.Figure([
        go.Scatter(x=y_test, y=y_pred, mode='markers', name="Predictions", marker=dict(color="blue")),
        go.Scatter(x=y_test, y=y_test, mode='lines', name="Ideal Line", line=dict(color='red', dash='dash'))
    ])
    fig.update_layout(title="Actual vs Predicted Values", xaxis_title="Actual Values", yaxis_title="Predicted Values", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # Residuals Distribution
    st.markdown("## 📊 Residuals Distribution")
    residuals = y_test - y_pred
    fig_residuals = px.histogram(residuals, nbins=20, title="Residuals Distribution", template="plotly_white")
    st.plotly_chart(fig_residuals, use_container_width=True)

else:
    st.sidebar.warning("⚠️ Please select features and target in the sidebar.")
    st.info("Select features and target from the sidebar to proceed!")
