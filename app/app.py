import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from pathlib import Path

from predictor import predict_activity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Human Activity Recognition",
        layout="wide"
)

# ============================================================
# CHIC UI THEME — VISUAL ONLY
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg: #E9EFF0;
        --surface: #F7F8F5;
        --surface-soft: #EEF3F1;
        --primary: #6E8F8A;
        --primary-dark: #587873;
        --secondary: #8C8AA8;
        --text: #2E3A42;
        --muted: #6F7C82;
        --border: #D5DEDC;
        --success: #6F987B;
        --warning: #B59462;
        --error: #B87979;
    }

    html, body, [class*="css"] {
        font-family: "DM Sans", "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(140, 138, 168, 0.10), transparent 28%),
            radial-gradient(circle at 100% 10%, rgba(110, 143, 138, 0.12), transparent 30%),
            var(--bg);
        color: var(--text);
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2.2rem;
        padding-bottom: 3.5rem;
    }

    /* Typography */
    h1, h2, h3 {
        color: var(--text) !important;
        font-family: "DM Sans", "Segoe UI", sans-serif !important;
    }

    h1 {
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.7px;
        margin-bottom: 0.2rem !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.3px;
    }

    h3 {
        font-size: 1.08rem !important;
        font-weight: 600 !important;
    }

    p, .stMarkdown, label {
        color: var(--text);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(239, 244, 243, 0.96);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] label {
        color: var(--muted) !important;
        font-weight: 500;
    }

    /* Cards / metrics */
    div[data-testid="stMetric"] {
        background: rgba(247, 248, 245, 0.92);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 18px rgba(46, 58, 66, 0.055);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
    }

    /* Buttons */
    .stButton > button {
        background: var(--primary);
        color: #F9FBFA !important;
        border: 1px solid var(--primary);
        border-radius: 10px;
        min-height: 2.65rem;
        font-family: "DM Sans", "Segoe UI", sans-serif;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(88, 120, 115, 0.16);
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
        color: #FFFFFF !important;
        transform: translateY(-1px);
        box-shadow: 0 5px 14px rgba(88, 120, 115, 0.22);
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 3px rgba(110, 143, 138, 0.20);
    }

    /* Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: var(--surface);
        border-color: var(--border);
        border-radius: 10px;
    }

    input {
        background: var(--surface) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    div[data-baseweb="select"] * {
        color: var(--text);
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        color: var(--text) !important;
    }

    /* Slider */
    div[data-baseweb="slider"] {
        color: var(--primary);
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(247, 248, 245, 0.82);
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    div[data-testid="stExpander"] summary {
        color: var(--text) !important;
        font-weight: 600;
    }

    /* Progress bars */
    div[data-testid="stProgress"] > div {
        background-color: #DCE6E3;
        border-radius: 99px;
    }

    div[data-testid="stProgress"] > div > div {
        background: var(--primary);
        border-radius: 99px;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border-width: 1px;
    }

    /* Divider */
    hr {
        border-color: var(--border);
        margin: 1.3rem 0;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* Links */
    a {
        color: var(--primary-dark) !important;
    }

    /* Plotly containers */
    div[data-testid="stPlotlyChart"] {
        background: rgba(247, 248, 245, 0.65);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.35rem;
        box-shadow: 0 4px 14px rgba(46, 58, 66, 0.035);
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD DATA
# ============================================================

test_data = np.load(
    PROJECT_ROOT / "data" / "processed" / "test.npz",
    allow_pickle=True
)

X_test = test_data["X"]
y_test = test_data["y"]


replay_data = np.load(
    PROJECT_ROOT / "data" / "processed" / "har_windows.npz",
    allow_pickle=True
)

X_replay = replay_data["X"]
y_replay = replay_data["y"]
user_ids = replay_data["user_ids"]
experiment_ids = replay_data["experiment_ids"]


# ============================================================
# TITLE
# ============================================================

st.title("Human Activity Recognition")
# st.subheader("IMU Sensor Data + Machine Learning")

st.write(
    "This application uses real IMU sensor data from the "
    "UCI Human Activity Recognition dataset and a trained "
    "Random Forest classifier."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Controls")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Single Window",
        "Live Replay"
    ]
)


# ============================================================
# SINGLE WINDOW MODE
# ============================================================

if mode == "Single Window":

    st.sidebar.subheader("Dataset Controls")

    sample_index = st.sidebar.number_input(
        "Select sensor window",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    window = X_test[sample_index]

    actual_activity = y_test[sample_index]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction, confidence, probabilities = predict_activity(
        window
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    st.markdown("## Activity Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Activity",
            prediction
        )

    with col2:
        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )

    with col3:
        st.metric(
            "Window",
            f"{sample_index + 1} / {len(X_test)}"
        )

    st.divider()

    # --------------------------------------------------------
    # Actual vs Predicted
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"Actual Activity: **{actual_activity}**"
        )

    with col2:

        if prediction == actual_activity:

            st.success(
                "Prediction: CORRECT"
            )

        else:

            st.error(
                "Prediction: INCORRECT"
            )

    # --------------------------------------------------------
    # Probability Distribution
    # --------------------------------------------------------

    st.subheader("Prediction Probabilities")

    probability_items = sorted(
        probabilities.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Top 3
    st.markdown("### Top Predictions")

    top_three = probability_items[:3]

    for activity, probability in top_three:

        st.write(
            f"**{activity}** — {probability:.2%}"
        )

        st.progress(float(probability))

    # All classes
    with st.expander("View all class probabilities"):

        for activity, probability in probability_items:

            st.write(
                f"**{activity}** — {probability:.2%}"
            )

            st.progress(float(probability))

    # --------------------------------------------------------
    # Sensor Visualization
    # --------------------------------------------------------

    st.subheader("IMU Sensor Signals")

    sensor_names = [
        "Acc_X",
        "Acc_Y",
        "Acc_Z",
        "Gyro_X",
        "Gyro_Y",
        "Gyro_Z"
    ]

    sensor_df = pd.DataFrame(
        window,
        columns=sensor_names
    )

    sensor_df["Sample"] = range(len(sensor_df))

    # --------------------------------------------------------
    # Accelerometer
    # --------------------------------------------------------

    st.markdown("### Accelerometer")

    fig_acc = go.Figure()

    acc_colors = {
        "Acc_X": "#611C35",
        "Acc_Y": "#2E5077",
        "Acc_Z": "#FFA630"
    }

    for axis in ["Acc_X", "Acc_Y", "Acc_Z"]:

        fig_acc.add_trace(
            go.Scatter(
                x=sensor_df["Sample"],
                y=sensor_df[axis],
                mode="lines",
                name=axis,
                line=dict(color=acc_colors[axis], width=2)
            )
        )

    fig_acc.update_layout(
        xaxis_title="Sample",
        yaxis_title="Acceleration",
        height=400,
        hovermode="x unified",
        paper_bgcolor="#F7F8F5",
        plot_bgcolor="#F7F8F5",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
        xaxis=dict(gridcolor="#DCE5E3"),
        yaxis=dict(gridcolor="#DCE5E3")
    )

    st.plotly_chart(
        fig_acc,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Gyroscope
    # --------------------------------------------------------

    st.markdown("### Gyroscope")

    fig_gyro = go.Figure()

    gyro_colors = {
        "Gyro_X": "#29524A",
        "Gyro_Y": "#312509",
        "Gyro_Z": "#8F3985"
    }

    for axis in ["Gyro_X", "Gyro_Y", "Gyro_Z"]:

        fig_gyro.add_trace(
            go.Scatter(
                x=sensor_df["Sample"],
                y=sensor_df[axis],
                mode="lines",
                name=axis,
                line=dict(color=gyro_colors[axis], width=2)
            )
        )

    fig_gyro.update_layout(
        xaxis_title="Sample",
        yaxis_title="Angular Velocity",
        height=400,
        hovermode="x unified",
        paper_bgcolor="#F7F8F5",
        plot_bgcolor="#F7F8F5",
        font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
        xaxis=dict(gridcolor="#DCE5E3"),
        yaxis=dict(gridcolor="#DCE5E3")
    )

    st.plotly_chart(
        fig_gyro,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Raw Data
    # --------------------------------------------------------

    with st.expander("View raw 128-sample sensor data"):

        st.dataframe(
            sensor_df,
            use_container_width=True
        )


# ============================================================
# LIVE REPLAY MODE
# ============================================================

else:

    st.sidebar.subheader("Replay Controls")

    # --------------------------------------------------------
    # Subject Selection
    # --------------------------------------------------------

    subjects = sorted(
        np.unique(user_ids).tolist()
    )

    subject_options = ["All Subjects"] + subjects

    selected_subject = st.sidebar.selectbox(
        "Subject",
        subject_options
    )

    # --------------------------------------------------------
    # Experiment Selection
    # --------------------------------------------------------

    experiments = sorted(
        np.unique(experiment_ids).tolist()
    )

    experiment_options = ["All Experiments"] + experiments

    selected_experiment = st.sidebar.selectbox(
        "Experiment",
        experiment_options
    )

    # --------------------------------------------------------
    # Speed
    # --------------------------------------------------------

    replay_speed = st.sidebar.slider(
        "Replay Speed",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Time between consecutive sensor windows."
    )

    # --------------------------------------------------------
    # Filter Dataset
    # --------------------------------------------------------

    mask = np.ones(len(X_replay), dtype=bool)

    if selected_subject != "All Subjects":

        mask &= (
            user_ids == selected_subject
        )

    if selected_experiment != "All Experiments":

        mask &= (
            experiment_ids == selected_experiment
        )

    replay_indices = np.where(mask)[0]

    # --------------------------------------------------------
    # No Data
    # --------------------------------------------------------

    if len(replay_indices) == 0:

        st.warning(
            "No sensor windows match the selected filters."
        )

        st.stop()

    # --------------------------------------------------------
    # Replay Information
    # --------------------------------------------------------

    st.markdown("## Live Sensor Replay")

    st.info(
        f"Replay contains **{len(replay_indices)} sensor windows**."
    )

    st.write(
        "Each window contains 128 samples from six IMU channels "
        "and is passed through the trained Random Forest model."
    )

    # --------------------------------------------------------
    # Start / Stop
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        start_replay = st.button(
            "Start Replay",
            use_container_width=True
        )

    with col2:

        stop_replay = st.button(
            "Stop Replay",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Initialize Session State
    # --------------------------------------------------------

    if "replay_running" not in st.session_state:

        st.session_state.replay_running = False

    if "replay_position" not in st.session_state:

        st.session_state.replay_position = 0

    # --------------------------------------------------------
    # Button Actions
    # --------------------------------------------------------

    if start_replay:

        st.session_state.replay_running = True
        st.session_state.replay_position = 0

    if stop_replay:

        st.session_state.replay_running = False
        st.session_state.replay_position = 0

    # --------------------------------------------------------
    # Replay Area
    # --------------------------------------------------------

    if st.session_state.replay_running:

        position = st.session_state.replay_position

        if position >= len(replay_indices):

            st.session_state.replay_running = False

            st.success(
                "Replay completed."
            )

            st.stop()

        current_index = replay_indices[position]

        window = X_replay[current_index]

        actual_activity = y_replay[current_index]

        current_subject = user_ids[current_index]

        current_experiment = experiment_ids[current_index]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction, confidence, probabilities = predict_activity(
            window
        )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        st.markdown("### Current Sensor Window")

        meta1, meta2, meta3, meta4 = st.columns(4)

        with meta1:

            st.metric(
                "Window",
                f"{position + 1} / {len(replay_indices)}"
            )

        with meta2:

            st.metric(
                "Subject",
                current_subject
            )

        with meta3:

            st.metric(
                "Experiment",
                current_experiment
            )

        with meta4:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )

        # ----------------------------------------------------
        # Prediction Result
        # ----------------------------------------------------

        st.markdown("### Model Prediction")

        pred_col, actual_col = st.columns(2)

        with pred_col:

            st.metric(
                "Predicted Activity",
                prediction
            )

        with actual_col:

            st.metric(
                "Actual Activity",
                actual_activity
            )

        if prediction == actual_activity:

            st.success(
                f"Correct Prediction — {prediction}"
            )

        else:

            st.error(
                f" Incorrect — Model predicted {prediction}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        progress = (
            position + 1
        ) / len(replay_indices)

        st.progress(
            progress
        )

        # ----------------------------------------------------
        # Probability Distribution
        # ----------------------------------------------------

        st.markdown("### Model Confidence")

        probability_items = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for activity, probability in probability_items:

            st.write(
                f"**{activity}** — {probability:.2%}"
            )

            st.progress(
                float(probability)
            )

        # ----------------------------------------------------
        # Sensor DataFrame
        # ----------------------------------------------------

        sensor_names = [
            "Acc_X",
            "Acc_Y",
            "Acc_Z",
            "Gyro_X",
            "Gyro_Y",
            "Gyro_Z"
        ]

        sensor_df = pd.DataFrame(
            window,
            columns=sensor_names
        )

        sensor_df["Sample"] = range(
            len(sensor_df)
        )

        # ----------------------------------------------------
        # Accelerometer
        # ----------------------------------------------------

        st.markdown("### Accelerometer")

        fig_acc = go.Figure()

        acc_colors = {
            "Acc_X": "#611C35",
            "Acc_Y": "#2E5077",
            "Acc_Z": "#FFA630"
        }

        for axis in [
            "Acc_X",
            "Acc_Y",
            "Acc_Z"
        ]:

            fig_acc.add_trace(
                go.Scatter(
                    x=sensor_df["Sample"],
                    y=sensor_df[axis],
                    mode="lines",
                    name=axis,
                    line=dict(color=acc_colors[axis], width=2)
                )
            )

        fig_acc.update_layout(
            xaxis_title="Sample",
            yaxis_title="Acceleration",
            height=350,
            hovermode="x unified",
            paper_bgcolor="#F7F8F5",
            plot_bgcolor="#F7F8F5",
            font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
            xaxis=dict(gridcolor="#DCE5E3"),
            yaxis=dict(gridcolor="#DCE5E3")
        )

        st.plotly_chart(
            fig_acc,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Gyroscope
        # ----------------------------------------------------

        st.markdown("### Gyroscope")

        fig_gyro = go.Figure()

        gyro_colors = {
            "Gyro_X": "#29524A",
            "Gyro_Y": "#312509",
            "Gyro_Z": "#8F3985"
        }

        for axis in [
            "Gyro_X",
            "Gyro_Y",
            "Gyro_Z"
        ]:

            fig_gyro.add_trace(
                go.Scatter(
                    x=sensor_df["Sample"],
                    y=sensor_df[axis],
                    mode="lines",
                    name=axis,
                    line=dict(color=gyro_colors[axis], width=2)
                )
            )

        fig_gyro.update_layout(
            xaxis_title="Sample",
            yaxis_title="Angular Velocity",
            height=350,
            hovermode="x unified",
            paper_bgcolor="#F7F8F5",
            plot_bgcolor="#F7F8F5",
            font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
            xaxis=dict(gridcolor="#DCE5E3"),
            yaxis=dict(gridcolor="#DCE5E3")
        )

        st.plotly_chart(
            fig_gyro,
            use_container_width=True
        )

        # ----------------------------------------------------
        # Raw Data
        # ----------------------------------------------------

        with st.expander(
            "View current 128-sample window"
        ):

            st.dataframe(
                sensor_df,
                use_container_width=True
            )

        # ----------------------------------------------------
        # Advance Replay
        # ----------------------------------------------------

        time.sleep(replay_speed)

        st.session_state.replay_position += 1

        st.rerun()

    else:

        st.markdown("## Ready to Replay")

        st.write(
            "Select a subject and experiment from the sidebar, "
            "then press **Start Replay** to simulate sequential "
            "IMU sensor activity recognition."
        )

        st.info(
            "For your presentation, select a specific subject "
            "and experiment so the replay represents one continuous "
            "sensor session."
        )
        
        
# import streamlit as st
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import time
# from pathlib import Path

# from predictor import predict_activity


# # ============================================================
# # PAGE CONFIGURATION
# # ============================================================

# st.set_page_config(
#     page_title="Human Activity Recognition",
#         layout="wide"
# )

# # ============================================================
# # CHIC UI THEME — VISUAL ONLY
# # ============================================================

# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

#     :root {
#         --bg: #E9EFF0;
#         --surface: #F7F8F5;
#         --surface-soft: #EEF3F1;
#         --primary: #6E8F8A;
#         --primary-dark: #587873;
#         --secondary: #8C8AA8;
#         --text: #2E3A42;
#         --muted: #6F7C82;
#         --border: #D5DEDC;
#         --success: #6F987B;
#         --warning: #B59462;
#         --error: #B87979;
#     }

#     html, body, [class*="css"] {
#         font-family: "DM Sans", "Segoe UI", sans-serif;
#     }

#     .stApp {
#         background:
#             radial-gradient(circle at 10% 0%, rgba(140, 138, 168, 0.10), transparent 28%),
#             radial-gradient(circle at 100% 10%, rgba(110, 143, 138, 0.12), transparent 30%),
#             var(--bg);
#         color: var(--text);
#     }

#     .main .block-container {
#         max-width: 1400px;
#         padding-top: 2.2rem;
#         padding-bottom: 3.5rem;
#     }

#     /* Typography */
#     h1, h2, h3 {
#         color: var(--text) !important;
#         font-family: "DM Sans", "Segoe UI", sans-serif !important;
#     }

#     h1 {
#         font-size: 2.25rem !important;
#         font-weight: 700 !important;
#         letter-spacing: -0.7px;
#         margin-bottom: 0.2rem !important;
#     }

#     h2 {
#         font-size: 1.45rem !important;
#         font-weight: 650 !important;
#         letter-spacing: -0.3px;
#     }

#     h3 {
#         font-size: 1.08rem !important;
#         font-weight: 600 !important;
#     }

#     p, .stMarkdown, label {
#         color: var(--text);
#     }

#     /* Sidebar */
#     section[data-testid="stSidebar"] {
#         background: rgba(239, 244, 243, 0.96);
#         border-right: 1px solid var(--border);
#     }

#     section[data-testid="stSidebar"] > div {
#         padding-top: 2rem;
#     }

#     section[data-testid="stSidebar"] h1,
#     section[data-testid="stSidebar"] h2,
#     section[data-testid="stSidebar"] h3 {
#         color: var(--text) !important;
#     }

#     section[data-testid="stSidebar"] label {
#         color: var(--muted) !important;
#         font-weight: 500;
#     }

#     /* Cards / metrics */
#     div[data-testid="stMetric"] {
#         background: rgba(247, 248, 245, 0.92);
#         border: 1px solid var(--border);
#         border-radius: 16px;
#         padding: 1rem 1.1rem;
#         box-shadow: 0 5px 18px rgba(46, 58, 66, 0.055);
#     }

#     div[data-testid="stMetricLabel"] {
#         color: var(--muted) !important;
#         font-size: 0.82rem !important;
#         font-weight: 500 !important;
#     }

#     div[data-testid="stMetricValue"] {
#         color: var(--text) !important;
#         font-weight: 700 !important;
#         letter-spacing: -0.3px;
#     }

#     /* Buttons */
#     .stButton > button {
#         background: var(--primary);
#         color: #F9FBFA !important;
#         border: 1px solid var(--primary);
#         border-radius: 10px;
#         min-height: 2.65rem;
#         font-family: "DM Sans", "Segoe UI", sans-serif;
#         font-weight: 600;
#         box-shadow: 0 3px 10px rgba(88, 120, 115, 0.16);
#         transition: all 0.18s ease;
#     }

#     .stButton > button:hover {
#         background: var(--primary-dark);
#         border-color: var(--primary-dark);
#         color: #FFFFFF !important;
#         transform: translateY(-1px);
#         box-shadow: 0 5px 14px rgba(88, 120, 115, 0.22);
#     }

#     .stButton > button:focus {
#         box-shadow: 0 0 0 3px rgba(110, 143, 138, 0.20);
#     }

#     /* Inputs */
#     div[data-baseweb="select"] > div,
#     div[data-baseweb="input"] > div {
#         background: var(--surface);
#         border-color: var(--border);
#         border-radius: 10px;
#     }

#     input {
#         background: var(--surface) !important;
#         color: var(--text) !important;
#         border-color: var(--border) !important;
#     }

#     div[data-baseweb="select"] * {
#         color: var(--text);
#     }

#     /* Radio buttons */
#     div[role="radiogroup"] label {
#         color: var(--text) !important;
#     }

#     /* Slider */
#     div[data-baseweb="slider"] {
#         color: var(--primary);
#     }

#     /* Expanders */
#     div[data-testid="stExpander"] {
#         background: rgba(247, 248, 245, 0.82);
#         border: 1px solid var(--border);
#         border-radius: 12px;
#     }

#     div[data-testid="stExpander"] summary {
#         color: var(--text) !important;
#         font-weight: 600;
#     }

#     /* Progress bars */
#     div[data-testid="stProgress"] > div {
#         background-color: #DCE6E3;
#         border-radius: 99px;
#     }

#     div[data-testid="stProgress"] > div > div {
#         background: var(--primary);
#         border-radius: 99px;
#     }

#     /* Alerts */
#     div[data-testid="stAlert"] {
#         border-radius: 12px;
#         border-width: 1px;
#     }

#     /* Divider */
#     hr {
#         border-color: var(--border);
#         margin: 1.3rem 0;
#     }

#     /* Dataframe */
#     div[data-testid="stDataFrame"] {
#         background: var(--surface);
#         border: 1px solid var(--border);
#         border-radius: 10px;
#         overflow: hidden;
#     }

#     /* Links */
#     a {
#         color: var(--primary-dark) !important;
#     }

#     /* Plotly containers */
#     div[data-testid="stPlotlyChart"] {
#         background: rgba(247, 248, 245, 0.65);
#         border: 1px solid var(--border);
#         border-radius: 14px;
#         padding: 0.35rem;
#         box-shadow: 0 4px 14px rgba(46, 58, 66, 0.035);
#     }

#     </style>
#     """,
#     unsafe_allow_html=True
# )



# # ============================================================
# # PROJECT PATH
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent


# # ============================================================
# # LOAD DATA
# # ============================================================

# test_data = np.load(
#     PROJECT_ROOT / "data" / "processed" / "test.npz",
#     allow_pickle=True
# )

# X_test = test_data["X"]
# y_test = test_data["y"]


# replay_data = np.load(
#     PROJECT_ROOT / "data" / "processed" / "har_windows.npz",
#     allow_pickle=True
# )

# X_replay = replay_data["X"]
# y_replay = replay_data["y"]
# user_ids = replay_data["user_ids"]
# experiment_ids = replay_data["experiment_ids"]


# # ============================================================
# # TITLE
# # ============================================================

# st.title("Human Activity Recognition")
# # st.subheader("IMU Sensor Data + Machine Learning")

# st.write(
#     "This application uses real IMU sensor data from the "
#     "UCI Human Activity Recognition dataset and a trained "
#     "Random Forest classifier."
# )


# # ============================================================
# # SIDEBAR
# # ============================================================

# st.sidebar.header("Controls")

# mode = st.sidebar.radio(
#     "Select Mode",
#     [
#         "Single Window",
#         "Live Replay"
#     ]
# )


# # ============================================================
# # SINGLE WINDOW MODE
# # ============================================================

# if mode == "Single Window":

#     st.sidebar.subheader("Dataset Controls")

#     sample_index = st.sidebar.number_input(
#         "Select sensor window",
#         min_value=0,
#         max_value=len(X_test) - 1,
#         value=0,
#         step=1
#     )

#     window = X_test[sample_index]

#     actual_activity = y_test[sample_index]

#     # --------------------------------------------------------
#     # Prediction
#     # --------------------------------------------------------

#     prediction, confidence, probabilities = predict_activity(
#         window
#     )

#     # --------------------------------------------------------
#     # Result
#     # --------------------------------------------------------

#     st.markdown("## Activity Prediction")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric(
#             "Predicted Activity",
#             prediction
#         )

#     with col2:
#         st.metric(
#             "Confidence",
#             f"{confidence:.2%}"
#         )

#     with col3:
#         st.metric(
#             "Window",
#             f"{sample_index + 1} / {len(X_test)}"
#         )

#     st.divider()

#     # --------------------------------------------------------
#     # Actual vs Predicted
#     # --------------------------------------------------------

#     col1, col2 = st.columns(2)

#     with col1:

#         st.info(
#             f"Actual Activity: **{actual_activity}**"
#         )

#     with col2:

#         if prediction == actual_activity:

#             st.success(
#                 "Prediction: CORRECT"
#             )

#         else:

#             st.error(
#                 "Prediction: INCORRECT"
#             )

#     # --------------------------------------------------------
#     # Probability Distribution
#     # --------------------------------------------------------

#     st.subheader("Prediction Probabilities")

#     probability_items = sorted(
#         probabilities.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     # Top 3
#     st.markdown("### Top Predictions")

#     top_three = probability_items[:3]

#     for activity, probability in top_three:

#         st.write(
#             f"**{activity}** — {probability:.2%}"
#         )

#         st.progress(float(probability))

#     # All classes
#     with st.expander("View all class probabilities"):

#         for activity, probability in probability_items:

#             st.write(
#                 f"**{activity}** — {probability:.2%}"
#             )

#             st.progress(float(probability))

#     # --------------------------------------------------------
#     # Sensor Visualization
#     # --------------------------------------------------------

#     st.subheader("IMU Sensor Signals")

#     sensor_names = [
#         "Acc_X",
#         "Acc_Y",
#         "Acc_Z",
#         "Gyro_X",
#         "Gyro_Y",
#         "Gyro_Z"
#     ]

#     sensor_df = pd.DataFrame(
#         window,
#         columns=sensor_names
#     )

#     sensor_df["Sample"] = range(len(sensor_df))

#     # --------------------------------------------------------
#     # Accelerometer
#     # --------------------------------------------------------

#     st.markdown("### Accelerometer")

#     fig_acc = go.Figure()

#     for axis in ["Acc_X", "Acc_Y", "Acc_Z"]:

#         fig_acc.add_trace(
#             go.Scatter(
#                 x=sensor_df["Sample"],
#                 y=sensor_df[axis],
#                 mode="lines",
#                 name=axis,
#                     line=dict(color="#7B9EBA", width=2)
#             )
#         )

#     fig_acc.update_layout(
#         xaxis_title="Sample",
#         yaxis_title="Acceleration",
#         height=400,
#         hovermode="x unified",
#         paper_bgcolor="#F7F8F5",
#         plot_bgcolor="#F7F8F5",
#         font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
#         xaxis=dict(gridcolor="#DCE5E3"),
#         yaxis=dict(gridcolor="#DCE5E3")
#     )

#     st.plotly_chart(
#         fig_acc,
#         use_container_width=True
#     )

#     # --------------------------------------------------------
#     # Gyroscope
#     # --------------------------------------------------------

#     st.markdown("### Gyroscope")

#     fig_gyro = go.Figure()

#     for axis in ["Gyro_X", "Gyro_Y", "Gyro_Z"]:

#         fig_gyro.add_trace(
#             go.Scatter(
#                 x=sensor_df["Sample"],
#                 y=sensor_df[axis],
#                 mode="lines",
#                 name=axis,
#                     line=dict(color="#7FA79E", width=2)
#             )
#         )

#     fig_gyro.update_layout(
#         xaxis_title="Sample",
#         yaxis_title="Angular Velocity",
#         height=400,
#         hovermode="x unified",
#         paper_bgcolor="#F7F8F5",
#         plot_bgcolor="#F7F8F5",
#         font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
#         xaxis=dict(gridcolor="#DCE5E3"),
#         yaxis=dict(gridcolor="#DCE5E3")
#     )

#     st.plotly_chart(
#         fig_gyro,
#         use_container_width=True
#     )

#     # --------------------------------------------------------
#     # Raw Data
#     # --------------------------------------------------------

#     with st.expander("View raw 128-sample sensor data"):

#         st.dataframe(
#             sensor_df,
#             use_container_width=True
#         )


# # ============================================================
# # LIVE REPLAY MODE
# # ============================================================

# else:

#     st.sidebar.subheader("Replay Controls")

#     # --------------------------------------------------------
#     # Subject Selection
#     # --------------------------------------------------------

#     subjects = sorted(
#         np.unique(user_ids).tolist()
#     )

#     subject_options = ["All Subjects"] + subjects

#     selected_subject = st.sidebar.selectbox(
#         "Subject",
#         subject_options
#     )

#     # --------------------------------------------------------
#     # Experiment Selection
#     # --------------------------------------------------------

#     experiments = sorted(
#         np.unique(experiment_ids).tolist()
#     )

#     experiment_options = ["All Experiments"] + experiments

#     selected_experiment = st.sidebar.selectbox(
#         "Experiment",
#         experiment_options
#     )

#     # --------------------------------------------------------
#     # Speed
#     # --------------------------------------------------------

#     replay_speed = st.sidebar.slider(
#         "Replay Speed",
#         min_value=0.1,
#         max_value=2.0,
#         value=0.5,
#         step=0.1,
#         help="Time between consecutive sensor windows."
#     )

#     # --------------------------------------------------------
#     # Filter Dataset
#     # --------------------------------------------------------

#     mask = np.ones(len(X_replay), dtype=bool)

#     if selected_subject != "All Subjects":

#         mask &= (
#             user_ids == selected_subject
#         )

#     if selected_experiment != "All Experiments":

#         mask &= (
#             experiment_ids == selected_experiment
#         )

#     replay_indices = np.where(mask)[0]

#     # --------------------------------------------------------
#     # No Data
#     # --------------------------------------------------------

#     if len(replay_indices) == 0:

#         st.warning(
#             "No sensor windows match the selected filters."
#         )

#         st.stop()

#     # --------------------------------------------------------
#     # Replay Information
#     # --------------------------------------------------------

#     st.markdown("## Live Sensor Replay")

#     st.info(
#         f"Replay contains **{len(replay_indices)} sensor windows**."
#     )

#     st.write(
#         "Each window contains 128 samples from six IMU channels "
#         "and is passed through the trained Random Forest model."
#     )

#     # --------------------------------------------------------
#     # Start / Stop
#     # --------------------------------------------------------

#     col1, col2 = st.columns(2)

#     with col1:

#         start_replay = st.button(
#             "Start Replay",
#             use_container_width=True
#         )

#     with col2:

#         stop_replay = st.button(
#             "Stop Replay",
#             use_container_width=True
#         )

#     # --------------------------------------------------------
#     # Initialize Session State
#     # --------------------------------------------------------

#     if "replay_running" not in st.session_state:

#         st.session_state.replay_running = False

#     if "replay_position" not in st.session_state:

#         st.session_state.replay_position = 0

#     # --------------------------------------------------------
#     # Button Actions
#     # --------------------------------------------------------

#     if start_replay:

#         st.session_state.replay_running = True
#         st.session_state.replay_position = 0

#     if stop_replay:

#         st.session_state.replay_running = False
#         st.session_state.replay_position = 0

#     # --------------------------------------------------------
#     # Replay Area
#     # --------------------------------------------------------

#     if st.session_state.replay_running:

#         position = st.session_state.replay_position

#         if position >= len(replay_indices):

#             st.session_state.replay_running = False

#             st.success(
#                 "Replay completed."
#             )

#             st.stop()

#         current_index = replay_indices[position]

#         window = X_replay[current_index]

#         actual_activity = y_replay[current_index]

#         current_subject = user_ids[current_index]

#         current_experiment = experiment_ids[current_index]

#         # ----------------------------------------------------
#         # Prediction
#         # ----------------------------------------------------

#         prediction, confidence, probabilities = predict_activity(
#             window
#         )

#         # ----------------------------------------------------
#         # Metadata
#         # ----------------------------------------------------

#         st.markdown("### Current Sensor Window")

#         meta1, meta2, meta3, meta4 = st.columns(4)

#         with meta1:

#             st.metric(
#                 "Window",
#                 f"{position + 1} / {len(replay_indices)}"
#             )

#         with meta2:

#             st.metric(
#                 "Subject",
#                 current_subject
#             )

#         with meta3:

#             st.metric(
#                 "Experiment",
#                 current_experiment
#             )

#         with meta4:

#             st.metric(
#                 "Confidence",
#                 f"{confidence:.2%}"
#             )

#         # ----------------------------------------------------
#         # Prediction Result
#         # ----------------------------------------------------

#         st.markdown("### Model Prediction")

#         pred_col, actual_col = st.columns(2)

#         with pred_col:

#             st.metric(
#                 "Predicted Activity",
#                 prediction
#             )

#         with actual_col:

#             st.metric(
#                 "Actual Activity",
#                 actual_activity
#             )

#         if prediction == actual_activity:

#             st.success(
#                 f"Correct Prediction — {prediction}"
#             )

#         else:

#             st.error(
#                 f" Incorrect — Model predicted {prediction}"
#             )

#         # ----------------------------------------------------
#         # Progress
#         # ----------------------------------------------------

#         progress = (
#             position + 1
#         ) / len(replay_indices)

#         st.progress(
#             progress
#         )

#         # ----------------------------------------------------
#         # Probability Distribution
#         # ----------------------------------------------------

#         st.markdown("### Model Confidence")

#         probability_items = sorted(
#             probabilities.items(),
#             key=lambda x: x[1],
#             reverse=True
#         )

#         for activity, probability in probability_items[:3]:

#             st.write(
#                 f"**{activity}** — {probability:.2%}"
#             )

#             st.progress(
#                 float(probability)
#             )

#         # ----------------------------------------------------
#         # Sensor DataFrame
#         # ----------------------------------------------------

#         sensor_names = [
#             "Acc_X",
#             "Acc_Y",
#             "Acc_Z",
#             "Gyro_X",
#             "Gyro_Y",
#             "Gyro_Z"
#         ]

#         sensor_df = pd.DataFrame(
#             window,
#             columns=sensor_names
#         )

#         sensor_df["Sample"] = range(
#             len(sensor_df)
#         )

#         # ----------------------------------------------------
#         # Accelerometer
#         # ----------------------------------------------------

#         st.markdown("### Accelerometer")

#         fig_acc = go.Figure()

#         for axis in [
#             "Acc_X",
#             "Acc_Y",
#             "Acc_Z"
#         ]:

#             fig_acc.add_trace(
#                 go.Scatter(
#                     x=sensor_df["Sample"],
#                     y=sensor_df[axis],
#                     mode="lines",
#                     name=axis
#                 )
#             )

#         fig_acc.update_layout(
#             xaxis_title="Sample",
#             yaxis_title="Acceleration",
#             height=350,
#             hovermode="x unified",
#             paper_bgcolor="#F7F8F5",
#             plot_bgcolor="#F7F8F5",
#             font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
#             xaxis=dict(gridcolor="#DCE5E3"),
#             yaxis=dict(gridcolor="#DCE5E3")
#         )

#         st.plotly_chart(
#             fig_acc,
#             use_container_width=True
#         )

#         # ----------------------------------------------------
#         # Gyroscope
#         # ----------------------------------------------------

#         st.markdown("### Gyroscope")

#         fig_gyro = go.Figure()

#         for axis in [
#             "Gyro_X",
#             "Gyro_Y",
#             "Gyro_Z"
#         ]:

#             fig_gyro.add_trace(
#                 go.Scatter(
#                     x=sensor_df["Sample"],
#                     y=sensor_df[axis],
#                     mode="lines",
#                     name=axis
#                 )
#             )

#         fig_gyro.update_layout(
#             xaxis_title="Sample",
#             yaxis_title="Angular Velocity",
#             height=350,
#             hovermode="x unified",
#             paper_bgcolor="#F7F8F5",
#             plot_bgcolor="#F7F8F5",
#             font=dict(family="DM Sans, Segoe UI, sans-serif", color="#2E3A42"),
#             xaxis=dict(gridcolor="#DCE5E3"),
#             yaxis=dict(gridcolor="#DCE5E3")
#         )

#         st.plotly_chart(
#             fig_gyro,
#             use_container_width=True
#         )

#         # ----------------------------------------------------
#         # Raw Data
#         # ----------------------------------------------------

#         with st.expander(
#             "View current 128-sample window"
#         ):

#             st.dataframe(
#                 sensor_df,
#                 use_container_width=True
#             )

#         # ----------------------------------------------------
#         # Advance Replay
#         # ----------------------------------------------------

#         time.sleep(replay_speed)

#         st.session_state.replay_position += 1

#         st.rerun()

#     else:

#         st.markdown("## Ready to Replay")

#         st.write(
#             "Select a subject and experiment from the sidebar, "
#             "then press **Start Replay** to simulate sequential "
#             "IMU sensor activity recognition."
#         )

#         st.info(
#             "For your presentation, select a specific subject "
#             "and experiment so the replay represents one continuous "
#             "sensor session."
#         )
