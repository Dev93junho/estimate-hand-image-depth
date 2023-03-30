import streamlit as st
import os
import sys
from main import main as run_main

st.title("Depth Estimation Model Training and Conversion")

apply_pruning = st.checkbox("Apply Pruning")
apply_quantization = st.checkbox("Apply Quantization")

convert_to_bin = st.checkbox("Convert to OpenVINO IR format (.bin/.xml)")
convert_to_onnx = st.checkbox("Convert to ONNX format")

if st.button("Run"):
    if not os.path.exists("models"):
        os.makedirs("models")

    sys.argv = [sys.argv[0]]
    if apply_pruning:
        sys.argv.append("--prune")
    if apply_quantization:
        sys.argv.append("--quantize")

    run_main()

    st.success("Model training and conversion completed.")
