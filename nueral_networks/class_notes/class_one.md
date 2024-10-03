### Artificial Intelligence (Introduction)

Artificial Intelligence is the ability of a machine to take input in the presence of knowledge reason and outputs or exhibits information. It is the ability of a machine to display human like intelligence.

From a behaviorist perspective the `turing test` could be used to describe artificial intelligence 

#### Timeline of AI 
- 1900 - 50s : Language theory
- 1920 - 30s: predicate calculus and propositional logic
- 1940 - 50s: Cybernetics (man machine communication)
- 1950s - Digital computers 

#### Examples of AI systems
- Robots
- Chess-playing program
- Voice recognition system
- Speech recognition system
- Grammar checker
- Pattern recognition
- Medial diagnosis
- System malfunction rectifier

#### AI philosophy

Since AI deals with knowledge and how a man reasons a large part of AI is dependent on philosophy.
- Research like `speech act theory` was developed by philosophers 
- Philosophers developed many logic like permission and obligation.
- Contributions like Kant's proof in pure reasoning that learning from experience was impossible without some sort of innate conceptual apparatus
- Another recent development is recognition of deep connections
between the AI task of understanding what sort of knowledge an
intelligent system requires and the older philosophical activities
of metaphysics.


#### Strong AI vs Weak AI

The main difference between weak AI and strong AI is that weak AI is limited to specific tasks, while strong AI aims to mimic human intelligence.

##### Weak AI
Also known as narrow AI, weak AI is designed to perform a specific task or set of tasks within a limited context. Weak AI is the most common type of AI today and can be used for tasks such as speech recognition, image processing, and navigation systems.

##### Strong AI
Also known as artificial general intelligence (AGI), strong AI aims to replicate human cognitive abilities and brain functions, including general problem-solving skills and common sense. Strong AI can perform a wide variety of tasks, learn new skills, and solve problems in ways similar to humans.

#### AI Agents 
An AI agent is Ai capable of interacting with its environment, collect data, and use the data to perform self-determined tasks to meet predetermined goals 

#### Machine Learning
- Supervised Learning (labelled input compared against output, data pattern is observed)(mostly 2D as planes are divided into one hyperplane)
- Unsupervised Learning (unlabelled input, it involved automatic analyzation of data) 
    - Hebbian Learning Rule: The Hebbian learning rule is a simple unsupervised learning rule that increases the strength of a connection between neurons when they fire simultaneously, the weights between these neurons are adjusted by a learning rate, between the pre and post synaptic neurons.
    - Competitive Learning: In CL, a network of artificial neurons compete to respond to a specific input. The goal is to increase the specialization of each node in the network.
    - Self Organizing maps: Self-organizing maps (SOMs) are an unsupervised form of Machine Learning that can be used to cluster data that has many features. The SOM not only clusters your data but also “maps” it on to a lower dimension (usually two dimensions) so that you can more easily visualize the clusters
- Reinforcement Learning:- Trial and Error form of learning, based on probabilistic model of sequential decision making, RL can be applied to large problems with the need for prior knowledge

#### Applications of Reinforcement Learning
- RL can optimize decision sequences to improve healthcare outcomes.
- RL can be used to maximize traffic flow efficiency at intersections controlled by traffic lights.
- Social media engagement can be seen as a form of RL driven by social rewards.


#### Neural Networks
- Modelled to the CNS in the human biological system
- A neural network is a computing model that has many units working together with no central control.
- The connections between the units have weights that are modified by some learning algorithm.

##### Classes of NN based on Architecture
- Feed forward Neural Networks (FNN): Information flows in one direction from input to output, without any feedback loops. Examples: Single-layer Perceptron, Multilayer Perceptron (MLP).

- Recurrent Neural Networks (RNN): Networks with feedback connections, allowing them to maintain a form of memory. Examples: Basic RNNs, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU).

- Convolutional Neural Networks (CNN): Networks that process grid-like data (such as images) using convolution operations. Primarily used in image and video processing.

- Modular Neural Networks (MNN): Composed of multiple sub-networks that operate independently on sub-problems and combine their outputs. Used for solving complex problems.

- Autoencoders: Networks designed to learn efficient representations (codings) of data, typically for the purpose of dimensionality reduction or generative tasks.

#### Development Process
- Data Collection
- Separate into training and test sets
- Define a network structure
- Select a learning algorithm
- Set parameter values
- Transform data into network inputs
- Train network until desired accuracy is achieved
- Test network

##### Back-Propagation Learning 
This is a learning algorithm that helps train neural network making them better at predictions, The process of forward pass, error calculation, backward pass, and weights update is repeated for multiple epochs until the network's performance is satisfactory
- Forward propagation: The input vector is applied to the network's sensory nodes, and the effect is propagated through the network, layer by layer. 
- Loss function: The error of the model's predictions is measured using a loss function. 
- Backward propagation: The error at the output is propagated back to the hidden units. 
- Gradient descent: The model weights are updated in the opposite direction of the gradient to reduce the error. 


The **gradient descent formula** is used to update the parameters of a function to minimize the loss (or cost) function. The formula involves taking a step in the direction of the negative gradient of the loss function.

Here is the general **gradient descent update rule** for parameter \( \mathbf{w} \):

$$[
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f(\mathbf{w}^{(t)})
]$$

### Explanation of Terms:
- $\mathbf{w}^{(t)}$: The vector of parameters (weights) at the current iteration $t$.
- $mathbf{w}^{(t+1)}$: The updated vector of parameters after the $t+1$-th iteration.
- $eta $: The **learning rate**, a positive scalar that determines the step size in the direction of the negative gradient.
- $\nabla f(\mathbf{w}^{(t)})$: The **gradient** of the loss (cost) function $ f(\mathbf{w})$ with respect to the parameters $\mathbf{w} $ at the current iteration $ t $. This represents the partial derivatives of the function with respect to each parameter.


#### Evolutionary Computing && Knowledge-based Systems

Evolutionary computing is a subfield of artificial intelligence (AI) inspired by biological evolution used to solve optimization and search problems. 

Knowledge-Based Systems in AI
Knowledge-based systems (KBS) are AI systems that utilize a rich repository of domain-specific knowledge to solve complex problems. These systems rely on expert knowledge encoded in the form of rules, ontologies, or logic to draw inferences, solve problems, or assist decision-making.