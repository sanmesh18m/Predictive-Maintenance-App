
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.title("🏭 Industrial Predictive Maintenance")
st.subheader("Machine Failure Prediction Dashboard")

st.write(
    "Enter machine sensor measurements to estimate possible machine failure."
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    model = joblib.load("improved_random_forest.pkl")
    threshold = joblib.load("failure_threshold.pkl")
    return model, threshold

try:
    model, threshold = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ---------------- HISTORY STORAGE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Project Information")
    st.write("Model: Random Forest")
    st.write("Dataset: AI4I 2020")
    st.write("Task: Binary Classification")
    st.write("Target: Machine Failure")

# ---------------- INPUTS ----------------

st.header("⚙️ Machine Sensor Measurements")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        value=298.1
    )

    speed = st.number_input(
        "Rotational Speed (rpm)",
        value=1500
    )

    tool_wear = st.number_input(
        "Tool Wear (min)",
        value=10
    )

with col2:
    process_temp = st.number_input(
        "Process Temperature (K)",
        value=308.6
    )

    torque = st.number_input(
        "Torque (Nm)",
        value=40.0
    )

# ---------------- PREDICTION ----------------

if st.button("Predict Machine Failure", use_container_width=True):

    input_data = pd.DataFrame([[
        air_temp,
        process_temp,
        speed,
        torque,
        tool_wear
    ]], columns=[
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ])

    failure_index = list(model.classes_).index(1)

    probability = model.predict_proba(
        input_data
    )[0][failure_index]

    prediction = int(probability >= threshold)

    # Save this prediction in session history
    st.session_state.history.append({
        "Air Temperature": air_temp,
        "Process Temperature": process_temp,
        "Rotational Speed": speed,
        "Torque": torque,
        "Tool Wear": tool_wear,
        "Failure Probability (%)": round(probability * 100, 2),
        "Prediction": "Failure" if prediction == 1 else "No Failure"
    })

    if prediction == 1:
        st.error("⚠️ Possible Machine Failure Detected!")
    else:
        st.success("✅ No Failure Predicted")

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "This is a model estimate, not a guarantee of actual machine condition."
    )

# ---------------- HISTORY TABLE ----------------

st.divider()
st.header("📋 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    total = len(history_df)
    failures = (history_df["Prediction"] == "Failure").sum()
    normal = total - failures

    # Summary metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Predictions", total)
    c2.metric("Failures Predicted", int(failures))
    c3.metric("No Failure", int(normal))

    # Display table
    st.subheader("Previous Predictions")
    st.dataframe(history_df, use_container_width=True)

    # Probability chart
    st.subheader("📈 Failure Probability Chart")

    chart_df = history_df[
        ["Failure Probability (%)"]
    ].copy()

    chart_df.index = range(1, len(chart_df) + 1)
    chart_df.index.name = "Prediction Number"

    st.line_chart(chart_df)

    # Clear history
    if st.button("Clear Prediction History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info(
        "No prediction history yet. Enter machine measurements "
        "and click Predict Machine Failure."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "College ML Project | AI4I 2020 Predictive Maintenance Dataset"
)
