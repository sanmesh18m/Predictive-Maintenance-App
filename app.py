
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
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

# ---------------- LOAD DATASET ----------------

@st.cache_data
def load_dataset():
    return pd.read_csv("ai4i2020.csv")

try:
    df = load_dataset()
except Exception as e:
    df = None

# ---------------- HISTORY STORAGE ----------------

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR NAVIGATION ----------------

st.sidebar.title("🏭 Navigation")

page = st.sidebar.radio(
    "Select Page",
    ["Home", "Machine Prediction", "Dataset Analysis"]
)

st.sidebar.divider()

st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Dataset:** AI4I 2020")
st.sidebar.write("**Task:** Binary Classification")
st.sidebar.write("**Target:** Machine Failure")


# ==================================================
# PAGE 1: HOME
# ==================================================

if page == "Home":

    st.title("🏭 Industrial Predictive Maintenance")

    st.subheader("Machine Failure Prediction Dashboard")

    st.write(
        "This project uses Machine Learning to estimate "
        "the possibility of industrial machine failure."
    )

    st.divider()

    st.header("📌 Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Machine Learning Model", "Random Forest")
    col2.metric("Dataset", "AI4I 2020")
    col3.metric("Problem Type", "Binary Classification")

    st.divider()

    st.header("⚙️ Input Features")

    st.write("""
    The model uses five machine operating measurements:

    1. Air Temperature (K)
    2. Process Temperature (K)
    3. Rotational Speed (rpm)
    4. Torque (Nm)
    5. Tool Wear (min)
    """)

    st.header("🎯 Project Objective")

    st.write(
        "The objective is to predict possible machine failure "
        "using sensor measurements. This can support predictive "
        "maintenance planning and help reduce unexpected downtime."
    )

    st.info(
        "Use the Machine Prediction page to test machine measurements "
        "and the Dataset Analysis page to explore the dataset."
    )


# ==================================================
# PAGE 2: MACHINE PREDICTION
# ==================================================

elif page == "Machine Prediction":

    st.title("⚙️ Machine Failure Prediction")

    st.write(
        "Enter the machine operating measurements below."
    )

    st.divider()

    # Machine inputs

    col1, col2 = st.columns(2)

    with col1:

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.1
        )

        speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=0,
            max_value=5000,
            value=1500
        )

    with col2:

        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=150.0,
            value=40.0
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=100
        )

    st.divider()

    # Prediction

    if st.button(
        "Predict Machine Failure",
        use_container_width=True
    ):

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

        # Save prediction history

        st.session_state.history.append({

            "Air Temperature": air_temp,
            "Process Temperature": process_temp,
            "Rotational Speed": speed,
            "Torque": torque,
            "Tool Wear": tool_wear,

            "Failure Probability (%)": round(
                probability * 100, 2
            ),

            "Prediction": (
                "Failure" if prediction == 1
                else "No Failure"
            )
        })

        st.subheader("📊 Prediction Result")

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
            "This is a model estimate, not a guarantee "
            "of actual machine condition."
        )

    # Prediction history

    st.divider()

    st.header("📋 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        total = len(history_df)

        failures = (
            history_df["Prediction"] == "Failure"
        ).sum()

        normal = total - failures

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Predictions", total)
        c2.metric("Failures Predicted", int(failures))
        c3.metric("No Failure", int(normal))

        st.subheader("Previous Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.subheader("📈 Failure Probability Chart")

        chart_df = history_df[
            ["Failure Probability (%)"]
        ].copy()

        chart_df.index = range(1, len(chart_df) + 1)

        chart_df.index.name = "Prediction Number"

        st.line_chart(chart_df)

        # Download history

        csv_data = history_df.to_csv(index=False)

        st.download_button(
            "Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No prediction history yet. Enter machine "
            "measurements and click Predict Machine Failure."
        )


# ==================================================
# PAGE 3: DATASET ANALYSIS
# ==================================================

elif page == "Dataset Analysis":

    st.title("📊 Dataset Analysis")

    if df is None:

        st.error(
            "Dataset file not found. Please check that "
            "ai4i2020.csv is present in the project folder."
        )

        st.stop()

    st.write(
        "Explore the AI4I 2020 Predictive Maintenance Dataset."
    )

    st.divider()

    # Dataset overview

    st.header("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Failure Records",
        int(df["Machine failure"].sum())
    )

    st.divider()

    # Dataset preview

    st.header("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # Statistical summary

    st.header("📈 Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.divider()

    # Failure distribution

    st.header("⚠️ Machine Failure Distribution")

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    sns.countplot(
        data=df,
        x="Machine failure",
        ax=ax1
    )

    ax1.set_title("Machine Failure Count")
    ax1.set_xlabel("Machine Failure (0 = No, 1 = Yes)")
    ax1.set_ylabel("Number of Records")

    st.pyplot(fig1)

    plt.close(fig1)

    # Feature distribution

    st.header("📊 Feature Distribution")

    feature = st.selectbox(
        "Select a feature",
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
    )

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        ax=ax2
    )

    ax2.set_title(f"Distribution of {feature}")

    st.pyplot(fig2)

    plt.close(fig2)

    # Correlation heatmap

    st.header("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include="number"
    )

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax3
    )

    ax3.set_title("Feature Correlation Heatmap")

    st.pyplot(fig3)

    plt.close(fig3)

    st.success("Dataset analysis completed successfully!")


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "College ML Project | AI4I 2020 Predictive Maintenance Dataset"
)
