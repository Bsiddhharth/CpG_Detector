import streamlit as st
import torch
import numpy as np
from CpGPredictor import CpGPredictor, prepare_data  # Import from your existing code
from torch import nn

# load the trained model
model = torch.load('cpG_predictor.pth')
model.eval()

#  preprocess input sequence
def preprocess_input(seq):
    # Convert input string to sequence of numbers
    dna_to_int = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
    return [dna_to_int.get(base, 0) for base in seq]


st.title('CpG Detector')
st.write('This app predicts the number of CpG sites in a given DNA sequence.')

# Input DNA sequence
user_input = st.text_area("Enter a DNA sequence (e.g., 'AGCTCGATCGCG'):")


if st.button('Predict'):
    if user_input:
        # Preprocess and make prediction
        input_sequence = preprocess_input(user_input)
        input_tensor = torch.tensor(input_sequence).unsqueeze(0)  #add batch dimension
        with torch.no_grad():
            prediction = model(input_tensor)
        st.write(f"Predicted number of CpG sites: {prediction.item():.2f}")
    else:
        st.write("Please enter a DNA sequence.")

