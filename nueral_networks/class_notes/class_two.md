#### **Introduction to Neural Networks**
**Learning Outcomes**
- Be able to describe the biological basis for Artificial Neural Networks
- Know some applications of Artificial Neural Networks (ANNs)
- Describe encoding schemas
- Understand the diff between supervise and unsupervised learning

##### Outline
- Back propagation neural network (NN)
- Neural Network design issues
- Backpropagation and Artificial Neural Network applications

#### Back propagation neural network (NN)
Backpropagation is a training mechanism. It’s the process by which weights in the network's architecture (whether feedforward, convolutional, or recurrent) are adjusted based on the gradient of the error signal. 

It occurs during the training phase, typically after a forward pass has been completed and the loss has been calculated.

##### Steps
- During the forward pass, input data is processed layer by layer through the network, and each layer applies weights and activation functions to compute outputs.

- Eventually, the network produces an output, which is compared to the actual target label using a loss function 

- After the forward pass, backpropagation is used to update the weights and biases in the network.

- It begins at the output layer and works its way backward through the hidden layers to the input layer. The goal is to minimize the error between the predicted output and the actual target label.

- Error Calculation: The loss or error from the output is propagated back through the network to calculate the gradients of the loss with respect to each weight using the chain rule of calculus.

#### Design issues on NN
When designing and training a neural network (NN), several critical aspects need to be addressed to ensure successful model development and performance

1. Topology of Neural Networks: The topology (or architecture) of a neural network refers to the structure and arrangement of the network's layers, neurons, and connections. it determines how well the network will be able to model the underlying patterns in the data. 
    - Layer Types:
    - Number of Layers (Depth):
    - Number of Neurons per Layer (Width):
    - Activation Functions:
    - Non-linear activation functions
    - Connectivity:

2.  Dataset: The quality, size, and preprocessing of the dataset directly impact the performance of the neural network.
    - Size of the Dataset:
    - Quality of Data: The data should be relevant, accurate, and clean. Noisy or incomplete data can lead to poor model performance
    - Data Distribution: The dataset should be representative of the problem domain. If the training data does not cover the full range of scenarios the model will not perform well.

3. Validation: Validation is the process of assessing how well the neural network performs on a dataset that was not used for training

## Neural Network 
- Computer System made up of several simple and highly interconnected processing elements
- Biologically inspired tring to simulate NN in human brain.
- These elements process information by responding to external stimulus
- Most problems which were not solvable by traditional approaches frequently can be solved using neural networks.

#### Components of a neural network

A neural network is composed of several key components. Here are the main components of a typical neural network:

A neural network is composed of several key components that work together to process input data, perform computations, and produce outputs. Here are the main components of a typical neural network:

1. **Neurons (Nodes)**
The basic units of a neural network. Each neuron receives input, processes it through an activation function, and passes the result to the next layer.
    - Input Neurons
    - Hidden Neurons
    - Output Neurons


2. **Weights**
Weights are the parameters that determine the strength of the connection between neurons. Each connection between neurons is associated with a weight, and these weights are adjusted during training to minimize the loss function.

    - **Weights Initialization**: The weights are initialized randomly at the start of training and are updated through learning algorithms like gradient descent.

3. **Bias**
Bias is an additional parameter in each neuron that allows the model to fit the data better by shifting the activation function. The bias term ensures that even when the input is zero, the neuron can still produce an output. It essentially adds flexibility to the model.

4. **Activation Function**
The activation function introduces non-linearity into the model, enabling the neural network to learn and represent complex relationships. Without activation functions, a neural network would be a simple linear model.
    - **ReLU (Rectified Linear Unit)**: outputting the input directly if it's positive, otherwise outputting zero.
    - **Sigmoid**:  outputs values between 0 and 1.
    - **Tanh (Hyperbolic Tangent)**: Similar to the sigmoid but outputs values between -1 and 1.
    - **Softmax**: It converts the output into a probability distribution.

5. **Loss Function (Cost Function)**
The loss function quantifies the difference between the predicted output and the true target values.    
- Mean Squared Error (MSE): Used for regression problems.
- Cross-Entropy Loss (Log Loss): Used for classification problems, including binary and multi-class classification.

6. **Optimizer**
The optimizer is responsible for updating the weights and biases during training to minimize the loss function. It defines how the model learns from the data.

    - **Gradient Descent**: updates the weights in the direction of the negative gradient of the loss function.
    - Variants include **Stochastic Gradient Descent (SGD)**, **Mini-batch Gradient Descent**, **Adam Optimizer**, etc.

7. **Forward Propagation**
This is the process of passing input data through the network, from the input layer to the output layer, to compute the predicted output. During forward propagation, the input is multiplied by weights, biases are added, and the result is passed through activation functions.

8. **Backpropagation**
Backpropagation is the process of computing gradients of the loss function with respect to each weight in the network, using the chain rule. It propagates the error backward from the output layer to the hidden layers and updates the weights accordingly using the optimizer. This is the core learning mechanism in neural networks.

9. **Learning Rate**
The learning rate is a hyperparameter that controls the size of the weight updates during training. A higher learning rate may cause faster learning but can also lead to overshooting the minimum of the loss function. A lower learning rate ensures more precise updates but might slow down convergence.

10. **Epochs and Iterations**
    - **Epoch**: An epoch refers to one complete pass of the entire training dataset through the network.
    - **Iteration**: In the context of mini-batch gradient descent, an iteration is one update of the network's weights after processing a mini-batch of data.

11. **Regularization**
Regularization techniques are used to prevent overfitting by penalizing complex models
    - **Dropout**: Randomly deactivates a fraction of neurons during each forward pass to prevent co-adaptation and reduce overfitting.


### ANN History (1943)

Here's an easy-to-remember timeline showing the history of Artificial Neural Networks (ANN):

### **Timeline: History of ANN**

- **1943**:  
  **McCulloch and Pitts**  
  - Demonstrated that networks of artificial binary-valued neurons can perform calculations.
  - Basis for the **Perceptron Neural Network**.

- **1949**:  
  **Hebb**  
  - Learning happens in a network of perceptrons.
  - Introduced **Hebb's Rule**, the precursor to backpropagation.
  
- **1956**:  
  **Rochester & Utley**  
  - First ANN computer simulation by Rochester.
  - Showed binary pattern classification using adaptive weights.

- **1957**:  
  **Rosenblatt**  
  - Introduced the **Perceptron Neural Network**.
  - Demonstrated optical pattern recognition using a 400-photocell array.

- **1962**:  
  **Widrow**  
  - Introduced **Adaline** (Adaptive Linear Neuron) and **Delta Learning Rule**.
  - Applied to weather forecasting, character, and speech recognition.

- **1969**:  
  **Minsky and Papert**  
  - Criticized Perceptrons for only solving linearly separable (LS) problems.
  - Led to decreased ANN research in the 1970s.

- **1986**:  
  **Rumelhart and McClelland**  
  - Developed the **Backpropagation (BP) Neural Network** with hidden layers.
  - Overcame Perceptron’s limitations to solve non-linearly separable (NLS) problems.

This timeline highlights the key breakthroughs and challenges in the development of ANN, leading up to the popularization of backpropagation in 1986.


## Perceptron: A Brief Overview

The **Perceptron** is one of the earliest and simplest types of artificial neural networks, developed by Frank Rosenblatt in 1957. It is a binary classifier that maps its input $ \mathbf{x}$ to an output value (0 or 1) based on a linear predictor function. The Perceptron consists of input nodes, weights, a bias, and an activation function, typically a step function.

#### **Mathematical Foundation**

1. **Inputs**: A vector of input values, $ \mathbf{x} = [x_1, x_2, \ldots, x_n] $.
2. **Weights**: A vector of weights, $ \mathbf{w} = [w_1, w_2, \ldots, w_n] $.
3. **Bias**: A bias term $ b $.
4. **Activation Function**: A step function defined as:

   $
   f(z) = 
   \begin{cases} 
   1 & \text{if } z \geq threshold \\
   0 & \text{if } z < threshold
   \end{cases}
   $

5. **Net Input Calculation**: (Hard Limit)

   $
   z = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b
   $

6. **Output Calculation**:

   $
   y = f(z)
   $

#### **Example: Solving a Perceptron Problem**

**Problem Statement**: Let's classify the following points:

| $ x_1 $ | $ x_2 $ | Class (Target) |
|-----------|-----------|-----------------|
| 0         | 0         | 0               |
| 0         | 1         | 0               |
| 1         | 0         | 0               |
| 1         | 1         | 1               |

We will train a perceptron to learn the AND function. ()

**Step 1: Initialize Weights and Bias**
- Set initial weights $ w_1 = 0.0 $, $ w_2 = 0.0 $, and bias $ b = 0.0 $.

**Step 2: Learning Rate**
- Set the learning rate $ \eta = 0.1 $.

**Step 3: Training**
- For each training sample, compute the output and update weights if the prediction is incorrect.

**Step 4: Training Iterations**

Assuming we run for a few epochs (iterations over the entire dataset):

1. For $ (0, 0) $, target = 0:
   - $ z = 0.0 \cdot 0 + 0.0 \cdot 0 + 0.0 = 0.0 $ ⇒ output = 1 (prediction is wrong)
   - Update weights:
     - $ w_1 = w_1 + \eta \cdot (target - output) \cdot x_1 = 0.0 + 0.1 \cdot (0 - 1) \cdot 0 = 0.0 $
     - $ w_2 = w_2 + \eta \cdot (target - output) \cdot x_2 = 0.0 + 0.1 \cdot (0 - 1) \cdot 0 = 0.0 $
     - $ b = b + \eta \cdot (target - output) = 0.0 + 0.1 \cdot (0 - 1) = -0.1 $

2. For $ (0, 1) $, target = 0:
   - $ z = 0.0 \cdot 0 + 0.0 \cdot 1 - 0.1 = -0.1 $ ⇒ output = 0 (prediction is correct)

3. For $ (1, 0) $, target = 0:
   - $ z = 0.0 \cdot 1 + 0.0 \cdot 0 - 0.1 = -0.1 $ ⇒ output = 0 (prediction is correct)

4. For $ (1, 1) $, target = 1:
   - $ z = 0.0 \cdot 1 + 0.0 \cdot 1 - 0.1 = -0.1 $ ⇒ output = 0 (prediction is wrong)
   - Update weights:
     - $ w_1 = w_1 + \eta \cdot (1 - 0) \cdot 1 = 0.0 + 0.1 \cdot 1 = 0.1 $
     - $ w_2 = w_2 + \eta \cdot (1 - 0) \cdot 1 = 0.0 + 0.1 \cdot 1 = 0.1 $
     - $ b = b + \eta \cdot (1 - 0) = -0.1 + 0.1 = 0.0 $

### **Final Weights After One Epoch**
After several epochs of this training process, the weights might converge to:
- $ w_1 = 0.5 $
- $ w_2 = 0.5 $
- $ b = -0.7 $

### **Pseudo Code for Perceptron Algorithm**

```plaintext
Initialize weights w and bias b to small random values
Set learning rate η
Set number of epochs E

For each epoch from 1 to E do:
    For each training example (x, target) do:
        z = w1 * x1 + w2 * x2 + b  // Compute net input
        output = step_function(z)    // Activation function

        // Update weights and bias if prediction is incorrect
        if output != target then
            for i from 1 to n do:
                w[i] = w[i] + η * (target - output) * x[i]
            b = b + η * (target - output)

Return the final weights and bias
```
