import streamlit as st
import numpy as np
from helpers import NeuronInitializer, McCullochPittsNeuron

def generate_binary_input(size):
    # Generate a binary input pattern of the specified size
    return np.random.choice([0, 1], size=size)

def plot_simulation(input_pattern, bias_weight, threshold):
    input_size = len(input_pattern) + 1  # Include bias
    neuron_initializer = NeuronInitializer()

    # Initialize a neuron with random weights
    neuron = neuron_initializer.initialize_neuron(input_size, bias_weight, threshold)

    # Activate the neuron
    output, weighted_sum = McCullochPittsNeuron.activate_neuron(neuron, input_pattern)

    # Display the result
    st.write("Input Pattern:", input_pattern)
    st.write("Bias Weight:", bias_weight)
    st.write("Threshold:", threshold)
    st.write("Weighted Sum:", weighted_sum)
    st.write("Output:", output)

def main():
    st.title("McCulloch-Pitts Neuron Simulation")

    # User input for the size of the input pattern
    input_size = st.slider("Input Pattern Size", min_value=1, max_value=10, value=5)

    # Generate a binary input pattern based on the user's chosen size
    input_pattern = generate_binary_input(input_size)

    # Interactive sliders for other parameters
    bias_weight = st.slider("Bias Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.01)


    # Button to run simulation
    if st.button("Run Simulation"):
        plot_simulation(input_pattern, bias_weight, threshold)
    st.write("-----------------------------------------------------------")
    st.write("- Based on selected size input patterns are randomly generated")
    st.write(
        "- At this point threshold is between 0 and 1. If required, you can try with multiple thresholds during training process")


if __name__ == "__main__":
    main()
