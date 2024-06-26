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


----------------------------------------------------------------------------------------------------------------------------------------

## **HW to Chapters 6  “Deep Neural Networks” & and 7 "Activation Functions"**

Deep Neural Network and Activation functions

Code available in branch -- "Deep_NN_&Activation_Functions"

**Details:**  

- network_classes.py : All classes for intializing parameters till forward propogation for depp neural network are available 
- main.py : Contains example input execution code
- back_prop.py : Contains back propogation function for layer to update weights and improve performance

To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example with required parameters


**Example: **

For the following parameters

input_dimensions = 2

num_of_samples = 10

hidden_layer_sizes = [3, 4]  # Two hidden layers with 3 and 4 neurons respectively

output_size = 1  #no of neurons in output layer

activation_types = ["relu", "sigmoid"]  # Activation functions for each layer


gives the following output:

<img width="950" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/df339bbd-e9c3-4ebd-9e1e-4ee85b5e5918">


----------------------------------------------------------------------------------------------------------------------------------------

## HW to Chapter 8 "Parameter Initialization and Training Sets"

Code Available in branch -- "training_datasets_develop"

**Details:**

- data_preparation_open_source.py: Creates train, test, val datasets from MNIST dataset, contains respective classes and functions
- data_prep_open_source.ipynb: Contains functions related to preparing open source data, can comfortably run in Google colab as libraries might raise compatibility issues on mac
- data_preparation_custom.py: Prepares custom image dataset with 70,20,10 images for train, test and val sets respectively, contains required classes and functions

  To execute the program:
  
- `pip install -r requirements.txt`
- `python data_preparation_custom.py` in terminal

Creates a folder named "data_open_source" with nested folders for train, test and val sets
  

----------------------------------------------------------------------------------------------------------------------------------------

## HW to Chapter 9 "Fitting, Bias, Regularization, and Dropout"

Code Available in branch -- "bias_regularization_and_dropout"

**Details:**

- network_classes.py: Contains all classes for forward propogation including drop out and regularization
  
- main.py: example input to runt the forward pass

  To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example with required parameters


----------------------------------------------------------------------------------------------------------------------------------------

## HW to Chapter 10 “Normalization and Optimization Methods

Code available in branch -- "input_normalization"

- network_classes.py has fucntions and classes related to normalization techniques
  
- main.py  user can chose the type of normalization technique they want to use in the forward pass

  To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example


----------------------------------------------------------------------------------------------------------------------------------------

## HW to Chapter 11 Learning Rates Decay and Hyperparameters
Code available in branch -- "Mini-batch"

**Details:**

- network_classes.py has classes related to mini batch process in forward pass

- main.py to simulate the process

   To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example


----------------------------------------------------------------------------------------------------------------------------------------
## HW to Chapter 12 “Softmax”
Code available in branch -- "Softmax"

**Details:**

- Activation_Classes.py: Contains softmax activation function and forward pass in neural network
  
- main.py: Simulation of Softmax Activation function with small input

   To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example

  ![image](https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/ee60e6fa-c2d4-4a35-971b-7f59c7b6e0a2)


----------------------------------------------------------------------------------------------------------------------------------------


## HW to Chapter 13 “Convolution Layer”
Code available in branch -- "CNN"

**Details:**

- convolution_layer.py: contains class with function performing convolution calculation, user can define input image matrix and kernel in main function

   To execute the program:

- `pip install -r requirements.txt`

- Run `python convolution_layer.py` for running a example

  ![image](https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/83d1ad17-4941-42e6-9880-2d0ad49bab55)



----------------------------------------------------------------------------------------------------------------------------------------

## HW to Chapter 15 "More convolutions, transfer learning
Code available in branch -- "Depth_Wise_CNN"


**Details:**

- Convolution_classes.py : classes for both depth wise and point wise convolution
- main.py : User can initialize image and choose between depth_wise and point_wise to calculate and visualize

  To execute the program:

- `pip install -r requirements.txt`

- Run `python main.py` for running a example

  **outputs**:

  Depth Wise :

  ![image](https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/e7e96ff3-47ca-4921-9df9-351678eb8ff1)

  <img width="895" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/8fd38f6e-6c05-4861-a053-9b851160f2ba">


  Point Wise:

  <img width="944" alt="image" src="https://github.com/Venkata-Bhargavi/Neural_Networks/assets/114631063/7d8e95f3-778a-4bd7-972d-55ecfe847783">

