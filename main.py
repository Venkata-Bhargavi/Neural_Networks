from helpers import NeuronInitializer, McCullochPittsNeuron

def run_simulation(input_pattern, bias_weight, threshold):
    input_size = len(input_pattern) + 1  # Include bias
    neuron_initializer = NeuronInitializer()

    # Initialize a neuron with random weights
    neuron = neuron_initializer.initialize_neuron(input_size, bias_weight, threshold)

    # Activate the neuron
    output = McCullochPittsNeuron.activate_neuron(neuron, input_pattern)

    # Display the result
    print("Input Pattern: ", input_pattern)
    print("Bias Weight: ", bias_weight)
    print("Threshold: ",threshold)
    print("Output: ", output)

if __name__ == "__main__":
    # Example usage in main.py
    input_pattern = [1, 0, 1, 1, 0]  # Excluding bias
    bias_weight = 0.5
    threshold = 0.0

    run_simulation(input_pattern, bias_weight, threshold)



