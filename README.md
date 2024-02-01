# Neural_Networks

To simulate mcculloch pitts neuron

Set up the environment and install all required libraries using `pip install -r requirements.txt`

In main.py change input patterns if required and run `python main.py` in terminal or run the main.py file in IDE like PyCharm

## Output:

<img width="995" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/5eb684ed-b29f-4233-b908-6d102b6f19bc">

Note: 
- Threshold is hard coded at this point but can be modified or randomly generated. If required, you can try with multiple thresholds during training process
- Bias input pattern is not mentioned explicitely as it is considered "1" in general and multiplying the weight with "1" would generate same result

## Streamlit app

The simulation is hosted on streamlit cloud and can be found in the following link
https://mccullochpittssimulatorvenkatabhargavi.streamlit.app/

To run the application in terminal use the following commands

`streamlit run mcculloch_pitts_simulator.py` and you will be able to launch the app on your local host


<img width="1000" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/1366eb58-a599-4189-a328-ab25ba8b66d3">
<img width="932" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/58bb30e9-389d-4fa7-8d2b-dab7324e75ac">

**Non Programming Assingment:**

[HW to Chapter 1 _Brain, Neurons, and Models_.pdf](https://github.com/Venkata-Bhargavi/Neural_Networks/files/13981044/HW.to.Chapter.1._Brain.Neurons.and.Models_.pdf)






----------------------------------------------------------------------------------------------------------------------------------------

## **HW to Chapter - 3 The Perceptron for Logistic Regression**


Code available in branch - "develop_HW3_perceptron_train"

**Details**:

All code files are available inside the directory `perceptron_for_logistic_regression` 

`helpers.py` : Classes and common functions used to initialize perceptrop, update weights, forward propogation activation function and creating image datasets

`dataset_preparation.py` : Uses the functions from helpers.py and store the images in _"dataset_handwritten_variations"_ and  _"test_images"_ folders. Creates train and test csv files as well

`main.py` : Initializes the perceptron and trains the perceptron using the dataset created from previous files

To execute the program

- `pip install -r requirements.txt`

- Run `python dataset_preparation.py` for creating train and test datasets

- Run `python main.py` for initializing perceptron and produting validation and test results


**Output:**

<img width="792" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/d0c0db40-186c-4f7d-98f9-d4f28fb37b2c">
<img width="1231" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/ae88f195-e1a6-4d5a-8b14-fdeebb14f33d">
<img width="1134" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/1776ccf1-fded-4fd2-8172-223eb5090d1f">
<img width="986" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/196916ad-b6c3-42b4-96a8-cf1bde0e5751">
<img width="698" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/2b730dee-8c8f-4a34-87e7-eeb50d9eaa10">


----------------------------------------------------------------------------------------------------------------------------------------

## **HW to Chapter - 4,5 Neural Network With One Hidden Layer**

Why do we need hidden layers when a perceptron with direct input and output can give us predictions?

Hidden layers in neural networks enable the learning of complex patterns and relationships within data by transforming inputs through nonlinear functions, facilitating the network's ability to generalize and make accurate predictions beyond simple linear mappings. They act as intermediate computational units that extract and process features from the input data, allowing for hierarchical representation learning and the modeling of intricate data structures.

Code available in branch -- `develop_HW_4_5_one_hidden_layer`

**Details:**

All code files are available in the directory `Neural_Network_One_Hidden_Layer`

`helpers.py` : Class for Activation functions and weights initializations including forward and backward propogations
`main.py` : Example input and output to train neural network

To execute the program

- `pip install -r requirements.txt`

- Run `python main.py` for initializing weights and training network
  

**Example 1:**

For input weights

X = np.array([[2, 3], [1, 2], [3, 2], [2, 1]]) # 4 data points with 2 features

y = np.array([[0], [0], [1], [1]]) # (4,1) ------ 4 target outputs

**Note:**
- epochs=**1000**
- learning_rate=**0.1**
- activation=**sigmoid**

Output:

<img width="836" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/8b4dd58b-82dc-4602-930a-9af2f1975b0e">




**Example 2:**

For input weights 

X = np.array([[2, 3, 1, 4], [1, 2, 3, 4], [3, 2, 1, 4]])  # Input data (3 samples, 4 features)

y = np.array([[0], [1], [1]])                            # Output data (3 samples, 1 output)


**Note:**
- epochs=**1000**
- learning_rate=**0.1**
- activation=**sigmoid**

Output:

<img width="937" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/8c58a541-f4f1-471d-837f-9385946461d1">

