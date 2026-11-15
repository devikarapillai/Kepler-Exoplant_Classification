import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Kepler Exoplanet Classifier", page_icon="🪐")


# Load the trained model and preprocessing files (loaded once and reused)
@st.cache_resource
def load_files():
    model = joblib.load('random_forest_model.pkl')
    imputer = joblib.load('imputer.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, imputer, feature_columns


model, imputer, feature_columns = load_files()

# Application title
st.title("🪐 NASA Kepler Exoplanet Classification")

st.write(
    "This application uses a trained Random Forest model "
    "to classify Kepler objects as CONFIRMED, CANDIDATE or FALSE POSITIVE."
)

st.caption(
    "Each box starts at a typical value from the training data (median). "
    "Change the values you know and leave the rest as they are."
)

st.subheader("Enter Kepler Object Measurements")

with st.expander("Orbital Measurements", expanded=True):

    koi_period = st.number_input(
        "Orbital Period (days)",
        min_value=0.0,
        value=9.9267,
        format="%.4f"
    )

    koi_period_err1 = st.number_input(
        "Orbital Period Upper Uncertainty",
        value=0.000036,
        format="%.6f",
        step=0.000001
    )

    koi_period_err2 = st.number_input(
        "Orbital Period Lower Uncertainty",
        value=-0.000036,
        format="%.6f",
        step=0.000001
    )

    koi_time0bk = st.number_input(
        "Transit Epoch (BKJD)",
        value=137.3966,
        format="%.4f"
    )

    koi_time0bk_err1 = st.number_input(
        "Transit Epoch Upper Uncertainty",
        value=0.0041,
        format="%.6f",
        step=0.0001
    )

    koi_time0bk_err2 = st.number_input(
        "Transit Epoch Lower Uncertainty",
        value=-0.0041,
        format="%.6f",
        step=0.0001
    )

with st.expander("Transit Measurements"):

    koi_impact = st.number_input(
        "Impact Parameter",
        min_value=0.0,
        value=0.5354,
        format="%.4f"
    )

    koi_impact_err1 = st.number_input(
        "Impact Parameter Upper Uncertainty",
        value=0.1925,
        format="%.4f"
    )

    koi_impact_err2 = st.number_input(
        "Impact Parameter Lower Uncertainty",
        value=-0.2060,
        format="%.4f"
    )

    koi_duration = st.number_input(
        "Transit Duration (hours)",
        min_value=0.0,
        value=3.812,
        format="%.4f"
    )

    koi_duration_err1 = st.number_input(
        "Transit Duration Upper Uncertainty",
        value=0.143,
        format="%.4f"
    )

    koi_duration_err2 = st.number_input(
        "Transit Duration Lower Uncertainty",
        value=-0.143,
        format="%.4f"
    )

    koi_depth = st.number_input(
        "Transit Depth (ppm)",
        min_value=0.0,
        value=421.8
    )

    koi_depth_err1 = st.number_input(
        "Transit Depth Upper Uncertainty",
        value=20.7
    )

    koi_depth_err2 = st.number_input(
        "Transit Depth Lower Uncertainty",
        value=-20.7
    )

with st.expander("Planet Measurements"):

    koi_prad = st.number_input(
        "Planet Radius (Earth radii)",
        min_value=0.0,
        value=2.41
    )

    koi_prad_err1 = st.number_input(
        "Planet Radius Upper Uncertainty",
        value=0.52
    )

    koi_prad_err2 = st.number_input(
        "Planet Radius Lower Uncertainty",
        value=-0.30
    )

    koi_teq = st.number_input(
        "Equilibrium Temperature (K)",
        min_value=0.0,
        value=874.0
    )

    koi_insol = st.number_input(
        "Insolation Flux (Earth flux)",
        min_value=0.0,
        value=138.74
    )

    koi_insol_err1 = st.number_input(
        "Insolation Flux Upper Uncertainty",
        value=71.27
    )

    koi_insol_err2 = st.number_input(
        "Insolation Flux Lower Uncertainty",
        value=-39.20
    )

    koi_model_snr = st.number_input(
        "Transit Signal-to-Noise Ratio",
        min_value=0.0,
        value=23.4
    )

with st.expander("Stellar Measurements"):

    koi_steff = st.number_input(
        "Stellar Effective Temperature (K)",
        min_value=0.0,
        value=5770.0
    )

    koi_steff_err1 = st.number_input(
        "Stellar Temperature Upper Uncertainty",
        value=157.0
    )

    koi_steff_err2 = st.number_input(
        "Stellar Temperature Lower Uncertainty",
        value=-160.0
    )

    koi_slogg = st.number_input(
        "Stellar Surface Gravity (log10 cm/s²)",
        value=4.438,
        format="%.3f"
    )

    koi_slogg_err1 = st.number_input(
        "Stellar Surface Gravity Upper Uncertainty",
        value=0.071,
        format="%.3f"
    )

    koi_slogg_err2 = st.number_input(
        "Stellar Surface Gravity Lower Uncertainty",
        value=-0.129,
        format="%.3f"
    )

    koi_srad = st.number_input(
        "Stellar Radius (Solar radii)",
        min_value=0.0,
        value=1.0,
        format="%.3f"
    )

    koi_srad_err1 = st.number_input(
        "Stellar Radius Upper Uncertainty",
        value=0.251,
        format="%.3f"
    )

    koi_srad_err2 = st.number_input(
        "Stellar Radius Lower Uncertainty",
        value=-0.111,
        format="%.3f"
    )

with st.expander("Position and Brightness"):

    ra = st.number_input(
        "Right Ascension (degrees)",
        value=292.2629,
        format="%.4f"
    )

    dec = st.number_input(
        "Declination (degrees)",
        value=43.6190,
        format="%.4f"
    )

    koi_kepmag = st.number_input(
        "Kepler Magnitude",
        min_value=0.0,
        value=14.529,
        format="%.3f"
    )

if st.button("Predict Exoplanet Classification", type="primary"):

    input_data = pd.DataFrame(
        [[
            koi_period,
            koi_period_err1,
            koi_period_err2,
            koi_time0bk,
            koi_time0bk_err1,
            koi_time0bk_err2,
            koi_impact,
            koi_impact_err1,
            koi_impact_err2,
            koi_duration,
            koi_duration_err1,
            koi_duration_err2,
            koi_depth,
            koi_depth_err1,
            koi_depth_err2,
            koi_prad,
            koi_prad_err1,
            koi_prad_err2,
            koi_teq,
            koi_insol,
            koi_insol_err1,
            koi_insol_err2,
            koi_model_snr,
            koi_steff,
            koi_steff_err1,
            koi_steff_err2,
            koi_slogg,
            koi_slogg_err1,
            koi_slogg_err2,
            koi_srad,
            koi_srad_err1,
            koi_srad_err2,
            ra,
            dec,
            koi_kepmag
        ]],
        columns=feature_columns
    )

    # Handle missing values (same preprocessing as training)
    input_data_imputed = imputer.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_imputed)[0]
    probabilities = model.predict_proba(input_data_imputed)[0]
    confidence = probabilities.max() * 100

    # Display prediction
    st.subheader("Prediction Result")

    if prediction == "CONFIRMED":
        st.success(f"CONFIRMED: likely a real exoplanet ({confidence:.1f}% confidence)")

    elif prediction == "CANDIDATE":
        st.warning(f"CANDIDATE: possible exoplanet, needs follow-up ({confidence:.1f}% confidence)")

    else:
        st.error(f"FALSE POSITIVE: likely not a planet ({confidence:.1f}% confidence)")

    # Display probability for each class
    st.write("Probability for each class:")

    probability_table = pd.DataFrame({
        "Class": model.classes_,
        "Probability (%)": (probabilities * 100).round(1)
    })

    st.dataframe(probability_table, hide_index=True)

    # Display input values
    with st.expander("Show input values"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}))