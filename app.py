import streamlit as st
import pandas as pd
import pickle
import os

# Optional: set working directory (update if needed)
os.chdir('/home/parthi/Git/Data Science/Machine Learning/Mushroom_Classification/')

# Load model + encoders
with open('model_with_encoders.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
label_encoders = bundle['label_encoders']
feature_columns = bundle['feature_columns']

# Streamlit Title
st.title("🍄 Mushroom Classification App")
st.write("Predict whether a mushroom is edible or poisonous based on its characteristics.")

# ---- FILE UPLOAD OR MANUAL INPUT ---- #
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select input method:", ["Manual Entry", "Upload CSV"])

# ---------- INPUT METHOD: MANUAL ---------- #
if input_method == "Manual Entry":
    st.subheader("Manual Input")

    # Use only the values seen during training
    cap_shape = st.selectbox("Cap Shape", label_encoders['cap-shape'].classes_)
    cap_surface = st.selectbox("Cap Surface", label_encoders['cap-surface'].classes_)
    cap_color = st.selectbox("Cap Color", label_encoders['cap-color'].classes_)
    bruises = st.selectbox("Bruises", label_encoders['bruises'].classes_)
    odor = st.selectbox("Odor", label_encoders['odor'].classes_)
    gill_attachment = st.selectbox("Gill Attachment", label_encoders['gill-attachment'].classes_)
    gill_spacing = st.selectbox("Gill Spacing", label_encoders['gill-spacing'].classes_)
    gill_size = st.selectbox("Gill Size", label_encoders['gill-size'].classes_)
    gill_color = st.selectbox("Gill Color", label_encoders['gill-color'].classes_)
    stalk_shape = st.selectbox("Stalk Shape", label_encoders['stalk-shape'].classes_)
    stalk_root = st.selectbox("Stalk Root", label_encoders['stalk-root'].classes_)
    stalk_surface_above_ring = st.selectbox("Stalk Surface Above Ring", label_encoders['stalk-surface-above-ring'].classes_)
    stalk_surface_below_ring = st.selectbox("Stalk Surface Below Ring", label_encoders['stalk-surface-below-ring'].classes_)
    stalk_color_above_ring = st.selectbox("Stalk Color Above Ring", label_encoders['stalk-color-above-ring'].classes_)
    stalk_color_below_ring = st.selectbox("Stalk Color Below Ring", label_encoders['stalk-color-below-ring'].classes_)
    veil_type = st.selectbox("Veil Type", label_encoders['veil-type'].classes_)
    veil_color = st.selectbox("Veil Color", label_encoders['veil-color'].classes_)
    ring_number = st.selectbox("Ring Number", label_encoders['ring-number'].classes_)
    ring_type = st.selectbox("Ring Type", label_encoders['ring-type'].classes_)
    spore_print_color = st.selectbox("Spore Print Color", label_encoders['spore-print-color'].classes_)
    population = st.selectbox("Population", label_encoders['population'].classes_)
    habitat = st.selectbox("Habitat", label_encoders['habitat'].classes_)

    # Build DataFrame
    input_data = [[
        cap_shape, cap_surface, cap_color, bruises, odor,
        gill_attachment, gill_spacing, gill_size, gill_color,
        stalk_shape, stalk_root, stalk_surface_above_ring,
        stalk_surface_below_ring, stalk_color_above_ring,
        stalk_color_below_ring, veil_type, veil_color, ring_number,
        ring_type, spore_print_color, population, habitat
    ]]
    input_df = pd.DataFrame(input_data, columns=feature_columns)

# ---------- INPUT METHOD: FILE UPLOAD ---------- #
elif input_method == "Upload CSV":
    st.subheader("Upload a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV with same feature columns", type=['csv'])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.write(input_df.head())

        # Validate columns
        if set(input_df.columns) != set(feature_columns):
            st.error("⚠️ Uploaded file must contain exactly these columns:")
            st.code(", ".join(feature_columns))
            st.stop()
    else:
        input_df = None

# ---------- PREDICT BUTTON (COMMON) ---------- #
if st.button("🔍 Predict"):
    if input_df is None:
        st.warning("Please upload a file or complete the manual inputs.")
        st.stop()

    # Encode using stored LabelEncoders
    try:
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
    except Exception as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    # Predict
    predictions = model.predict(input_df)
    result_labels = ['🟢 Edible' if p == 0 else '🔴 Poisonous' for p in predictions]

    input_df['Prediction'] = result_labels

    st.success("🎯 Prediction Results")
    st.dataframe(input_df[['Prediction']])

    # Optional download
    csv = input_df.to_csv(index=False)
    st.download_button("📥 Download Result CSV", csv, file_name="mushroom_predictions.csv")

    # Count predictions
    num_predictions = len(predictions)
    num_edible = sum(pred == 0 for pred in predictions)
    num_poisonous = sum(pred == 1 for pred in predictions)

    # Percentage
    pct_edible = round((num_edible / num_predictions) * 100, 2)
    pct_poisonous = round((num_poisonous / num_predictions) * 100, 2)

    # Show metrics
    st.subheader("📊 Prediction Summary")
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", num_predictions)
    col2.metric("🟢 Edible", f"{num_edible} ({pct_edible}%)")
    col3.metric("🔴 Poisonous", f"{num_poisonous} ({pct_poisonous}%)")
