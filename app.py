
import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing files
model = joblib.load('random_forest_model.pkl')
imputer = joblib.load('imputer.pkl')
feature_columns = joblib.load('feature_columns.pkl')
st.write("Model classes:", model.classes_)

# Application title
st.title("NASA Kepler Exoplanet Classification")

st.write(
    "This application uses a trained Random Forest model "
    "to classify Kepler objects into three categories."
)

st.subheader("Enter Kepler Object Measurements")

with st.expander("Orbital Measurements", expanded=True):

    koi_period = st.number_input(
        "Orbital Period (days)",
        min_value=0.0,
        value=10.0
    )

    koi_period_err1 = st.number_input(
        "Orbital Period Error 1",
        value=0.0,
         format="%.4f",
         step=0.0001
    )

    koi_period_err2 = st.number_input(
        "Orbital Period Error 2",
        value=0.0,
        format="%.4f",
        step=0.0001
    )

    koi_time0bk = st.number_input(
        "Transit Epoch (BKJD)",
        value=0.0
    )

    koi_time0bk_err1 = st.number_input(
        "Transit Epoch Error 1",
        value=0.0,
        format="%.4f",
        step=0.0001
    )

    koi_time0bk_err2 = st.number_input(
        "Transit Epoch Error 2",
        value=0.0,
        format="%.4f",
        step=0.0001
    )

with st.expander("Transit Measurements"):

    koi_impact = st.number_input(
        "Impact Parameter",
        min_value=0.0,
        value=0.5
    )

    koi_impact_err1 = st.number_input(
        "Impact Parameter Error 1",
        value=0.0
    )

    koi_impact_err2 = st.number_input(
        "Impact Parameter Error 2",
        value=0.0
    )

    koi_duration = st.number_input(
        "Transit Duration (hours)",
        min_value=0.0,
        value=3.0
    )

    koi_duration_err1 = st.number_input(
        "Transit Duration Error 1",
        value=0.0
    )

    koi_duration_err2 = st.number_input(
        "Transit Duration Error 2",
        value=0.0
    )

    koi_depth = st.number_input(
        "Transit Depth (ppm)",
        min_value=0.0,
        value=400.0
    )

    koi_depth_err1 = st.number_input(
        "Transit Depth Error 1",
        value=0.0
    )

    koi_depth_err2 = st.number_input(
        "Transit Depth Error 2",
        value=0.0
    )

with st.expander("Planet Measurements"):

    koi_prad = st.number_input(
        "Planet Radius (Earth radii)",
        min_value=0.0,
        value=2.0
    )

    koi_prad_err1 = st.number_input(
        "Planet Radius Error 1",
        value=0.0
    )

    koi_prad_err2 = st.number_input(
        "Planet Radius Error 2",
        value=0.0
    )

    koi_teq = st.number_input(
        "Equilibrium Temperature (K)",
        min_value=0.0,
        value=500.0
    )

    koi_insol = st.number_input(
        "Insolation Flux",
        min_value=0.0,
        value=1.0
    )

    koi_insol_err1 = st.number_input(
        "Insolation Flux Error 1",
        value=0.0
    )

    koi_insol_err2 = st.number_input(
        "Insolation Flux Error 2",
        value=0.0
    )

    koi_model_snr = st.number_input(
        "Model Signal-to-Noise Ratio",
        min_value=0.0,
        value=20.0
    )

with st.expander("Stellar Measurements"):

    koi_steff = st.number_input(
        "Stellar Effective Temperature (K)",
        min_value=0.0,
        value=5500.0
    )

    koi_steff_err1 = st.number_input(
        "Stellar Temperature Error 1",
        value=0.0
    )

    koi_steff_err2 = st.number_input(
        "Stellar Temperature Error 2",
        value=0.0
    )

    koi_slogg = st.number_input(
        "Stellar Surface Gravity",
        value=4.5
    )

    koi_slogg_err1 = st.number_input(
        "Stellar Surface Gravity Error 1",
        value=0.0
    )

    koi_slogg_err2 = st.number_input(
        "Stellar Surface Gravity Error 2",
        value=0.0
    )

    koi_srad = st.number_input(
        "Stellar Radius (Solar radii)",
        min_value=0.0,
        value=1.0
    )

    koi_srad_err1 = st.number_input(
        "Stellar Radius Error 1",
        value=0.0
    )

    koi_srad_err2 = st.number_input(
        "Stellar Radius Error 2",
        value=0.0
    )

with st.expander("Other Measurements"):

    ra = st.number_input(
        "Right Ascension (degrees)",
        value=290.0
    )

    dec = st.number_input(
        "Declination (degrees)",
        value=45.0
    )

    koi_kepmag = st.number_input(
        "Kepler Magnitude",
        min_value=0.0,
        value=14.0
    )

if st.button("Predict Exoplanet Classification"):

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

 
    # Handle missing values
    input_data_imputed = imputer.transform(input_data)

# Make prediction
    prediction = model.predict(input_data_imputed)

   

# Display input and prediction
    st.write("Input values:", input_data)
    st.write("Predicted class:", prediction[0])

    # Display prediction
    st.subheader("Prediction Result")

    if prediction[0] == "CONFIRMED":
         st.success("CONFIRMED: The model predicts this object as confirmed.")

    elif prediction[0] == "CANDIDATE":
        st.warning("CANDIDATE: The model predicts this object as a candidate.")

    else:
        st.error("FALSE POSITIVE: The model predicts this object as a false positive.")
 