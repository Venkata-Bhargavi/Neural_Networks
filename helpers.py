import numpy as np

class NeuronInitializer:
    #you can call this method on the class itself with out creating the instance of the class
    @staticmethod
    def initialize_neuron(input_size, bias_weight, threshold):
        '''
        function to get initialized weights and return weights and threshold
        :param input_size: length of input pattern
        :param bias_weight: bias weight
        :param threshold: threshold value
        :return: K,V with initialized weights and threshold
        '''
        weights = NeuronInitializer.initialize_weights(input_size, bias_weight)
        return {'weights': weights, 'threshold': threshold}

    @staticmethod
    def initialize_weights(size, bias_weight):
        '''
        function to initialize weights by generating random numbers in given size of input patterns
        :param size: input pattern size
        :param bias_weight: weight of bias
        :return: randomized weights with added bias
        '''
        weights = np.random.rand(size)
        weights[-1] = bias_weight  # Set the last element as bias weight
        return weights

class McCullochPittsNeuron:
    @staticmethod
    def activate_neuron(neuron, inputs):
        '''
        function to calculate weighted sum and return output 1 if the sum is > threshold else 0
        :param neuron: K,V pair
        :param inputs: input patterns as list
        :return: 1 0r 0
        '''
        weights = neuron['weights']
        threshold = neuron['threshold']

        if len(inputs) != len(weights) - 1:
            raise ValueError("Number of inputs must match the number of weights (excluding bias)")

        weighted_sum = np.dot(weights[:-1], inputs) + weights[-1]  # Directly add bias term to the weighted sum
        output = 1 if weighted_sum >= threshold else 0
        return output



