### MSc Lecture Notes: The Recursive Deterministic Perceptron Growing Neural Network (RDP)

#### Outline:
- Linear Separability
    - Notion
    - Existing methods
    - Novel methods
    - Heuristics
- The Recursive Deterministic Perceptron (RDP) Growing Neural Network
    - Methods for building RDP neural networks.
    - Genetic Algorithm approach for building RDP neural networks.
    - The implementation and testing of a m-class RDP method for more classification problems
    - with more than 2 classes.

#### Preliminaries: 
1. Traditional Perceptron: Introduced by McCulloch and Pitts, a neural network that computes the weighted sum of input patterns and compares it to a threshold. Used for linearly separable (LS) data​. 
- Can only solve problems that are linearly separable 
- Limited to two-class classification problems 
- Uses the perceptron learning rule for weight updates

Recursive Deterministic Perceptron (RDP): A specialized multilayer network designed to handle both linearly separable (LS) and non-linearly separable (NLS) data.
- Utilizes a recursive structure to create additional layers as needed
- Can solve more complex problems than the traditional perceptron
- Does not require back propagation for training
- Automatically determines the number of hidden layers and neurons required

2. Linear Separability
Definition: Two classes are linearly separable if a hyperplane can separate them, placing each class on opposite sides of the hyperplane.
Testing Methods:
- Fourier-Kuhn Elimination and Simplex Algorithms: Solve systems of linear equations​.
- Computational Geometry Techniques: Use the Convex Hull method and the Class of Linear Separability method.
- Neural Network Approach: The Perceptron learning algorithm ensures convergence if the classes are LS.
- Quadratic Programming: Solves a quadratic optimization problem to find a separating hyperplane, as in the case of support vector machines.

### Solving Systems of Linear Equations
1. Solving Systems of Linear Equations
    - Fourier-Kuhn Elimination Algorithm: This algorithm eliminates variables from systems of linear inequalities to solve the linear separability problem. The algorithm iteratively eliminates variables from inequalities by combining them. If, at any point, the system produces a contradiction (e.g., two inequalities that cannot be simultaneously satisfied), the data is deemed non-linearly separable (NLS).

        ```md
        # Fourier-Kuhn Elimination Algorithm Example

        Let's consider a simple system of linear inequalities with three variables (x, y, z):

        1. x + y + z ≤ 10
        2. 2x - y + z ≤ 20
        3. -x + 2y - z ≤ 5
        4. x - y + 2z ≥ 4

        Our goal is to eliminate the variable z and project the system onto the xy-plane.

        ## Step 1: Rearrange inequalities
        First, we'll rearrange the inequalities so that z is isolated on one side:

        1. z ≤ 10 - x - y
        2. z ≤ 20 - 2x + y
        3. -z ≤ 5 + x - 2y
        4. z ≥ 2 - 1/2x + 1/2y

        ## Step 2: Separate inequalities
        Group the inequalities based on the sign of z's coefficient:

        Positive z:
        - z ≤ 10 - x - y
        - z ≤ 20 - 2x + y

        Negative z:
        - z ≥ 2 - 1/2x + 1/2y

        No z:
        - -z ≤ 5 + x - 2y (we'll convert this to: z ≥ -5 - x + 2y)

        ## Step 3: Combine inequalities
        Create new inequalities by combining each inequality from the positive z group with each from the negative z group:

        (10 - x - y) ≥ (2 - 1/2x + 1/2y)
        8 ≥ 1/2x - 3/2y

        (10 - x - y) ≥ (-5 - x + 2y)
        15 ≥ 3y

        (20 - 2x + y) ≥ (2 - 1/2x + 1/2y)
        18 ≥ 3/2x

        (20 - 2x + y) ≥ (-5 - x + 2y)
        25 ≥ x + y

        ## Step 4: Simplify and combine results
        Our new system of inequalities without z is:

        1. x - 3y ≤ 16
        2. y ≤ 5
        3. x ≤ 12
        4. x + y ≤ 25

        This system describes the projection of our original 3D system onto the xy-plane.
        ```

    - Simplex Algorithm: The Simplex algorithm is widely used in linear programming to find the optimal solution to linear equations and inequalities.

    The algorithm optimizes a function, adjusting the solution iteratively by pivoting through different solutions in the feasible region until an optimal separating hyperplane is found.

2. Methods Based on Computational Geometry Techniques:

    These methods utilize geometrical properties of data points and their arrangements in space to determine linear separability. The two key methods discussed are the Convex Hull method and the Class of Linear Separability (CLS) method.

    - Convex Hull Method: The Convex Hull method uses the concept of convex hulls, which are the smallest convex sets that enclose all data points in a class.

        Construct the convex hulls for both classes of data points.
        Check if the convex hulls of the two classes intersect.

        If the convex hulls intersect, the data is non-linearly separable (NLS); if not, the data is linearly separable (LS).

    - Class of Linear Separability (CLS) Method
        Concept: The CLS method characterizes the points in space by which a hyperplane passes that linearly separates two classes.

        The method iteratively identifies the points that lie on the boundary of separability between two classes.
        Once the separating hyperplane is found, it divides the data into two classes. If no such hyperplane can be found, the data is NLS.

3. Methods Based on Neural Networks: 
Neural networks, particularly perceptrons, are a natural fit for testing linear separability. The perceptron learning algorithm is designed to converge if a dataset is linearly separable.

    - Perceptron Learning Algorithm: The perceptron is the simplest form of a neural network and is capable of solving linearly separable problems.

        Initialize the perceptron with random weights.
        For each data point, the perceptron computes the weighted sum of inputs and compares it to a threshold.
        If the perceptron misclassifies a point, it updates the weights to better classify the point.
        The algorithm repeats until all points are correctly classified or a maximum number of iterations is reached.

4. Methods Based on Quadratic Programming

    - Support Vector Machine (SVM)
        
        **Concept**:

        Support Vector Machines are powerful supervised learning models used for classification and regression tasks. The key idea behind SVMs is to find the optimal hyperplane that best separates different classes of data points while maximizing the margin between them.
        Let's break this down:

        Hyperplane: In a 2D space, this would be a line. In 3D, it's a plane. In higher dimensions, it's called a hyperplane.
        Optimal: The hyperplane that provides the best separation between classes.
        Margin: The distance between the hyperplane and the nearest data point from either class. SVMs aim to maximize this margin.

        **Procedure**:

        Formulating the classification problem as a quadratic optimization problem (QPOP):

        The goal is to find the hyperplane that maximizes the margin while minimizing classification errors.
        This is expressed mathematically as an optimization problem where we want to minimize a quadratic function subject to linear constraints.
        The objective function includes terms for maximizing the margin and minimizing the classification error.

        Handling non-linearly separable data:

        Real-world data is often not linearly separable in its original space.
        SVMs use a technique called the "kernel trick" to project the data into a higher-dimensional space.
        In this higher-dimensional space, the data becomes linearly separable.
        Common kernel functions include:

        - Linear kernel
        - Polynomial kernel
        - Radial Basis Function (RBF) kernel
        - Sigmoid kernel

5. Fisher Linear Discriminant (FLD) Method

    The FLD method is another technique for linear separability testing, which focuses on maximizing the distance between class projections while minimizing variance within classes.

    Procedure:
    Compute the within-class and between-class scatter matrices.
    Solve for the weight vector that maximizes the ratio of the between-class scatter to the within-class scatter.
    Use this weight vector to find the hyperplane that best separates the classes.

###  Linear Separability in Recursive Deterministic Perceptron (RDP)

1. Importance of Linear Separability in RDP: Linear separability is a foundational concept in neural networks, especially for the Recursive Deterministic Perceptron (RDP). A problem is considered linearly separable (LS) if there exists a hyperplane in the feature space that can completely separate the data points of different classes. However, many real-world classification problems are non-linearly separable (NLS), meaning a single linear hyperplane cannot distinguish between all data points of different classes.

    RDP uses a divide-and-conquer strategy, where subsets of the NLS data that are linearly separable are identified. The core idea is to progressively transform the original NLS problem into smaller LS problems by adding dimensions and intermediate layers to the input vector, allowing it to become LS in the new augmented space.

    The Recursive Deterministic Perceptron (RDP) is a **multilayer neural network** that addresses **non-linearly separable (NLS) problems** by transforming them into **linearly separable (LS) subsets** through recursive operations. This network is a generalized version of the **single-layer perceptron** and solves complex classification tasks by dynamically adding layers and dimensions to the input space. Below, we’ll explore the mathematical foundations, construction principles, characteristics, and an example to solidify the understanding of how RDP works.

---

### 1. **Mathematical Foundation of RDP**

#### 1.1 **Recursive Construction and Notation**
In an NLS dataset, the data points of two or more classes cannot be separated by a single hyperplane in the given space. The RDP transforms the input space by recursively adding layers. Each added layer is associated with a hyperplane that separates a **linearly separable subset (LS subset)** from the rest of the dataset.

- Let the input dataset $ X = \{x_1, x_2, ..., x_n\} $ with $ x_i \in \mathbb{R}^d $  be NLS.
- The goal of the RDP is to **augment the input space** such that it becomes linearly separable by adding recursive dimensions.

#### 1.2 **Input Augmentation and Linear Separation**
- Consider an input vector $ x \in \mathbb{R}^d $ . The recursive construction adds a new component, $ h $ , to the input vector that represents the output of an intermediate neuron corresponding to a linear separation of an LS subset:
  \[
  x' = [x, h] \quad \text{where} \quad h = w^T x + b
  \]
  where $ w \in \mathbb{R}^d $  is the weight vector that defines the separating hyperplane for the LS subset, and $ b $  is the bias term.
  
- **Augmented Input Vector**: After adding a new dimension for the LS subset, the input vector becomes $ x' \in \mathbb{R}^{d+1} $ , allowing a new hyperplane to be found for this augmented space.
  
- **Recursive Separation**: The process continues recursively, with each added dimension allowing further separation until all points are separated or the NLS problem becomes LS.

---

### 2. **Characteristics of RDP**

The Recursive Deterministic Perceptron has several key characteristics that differentiate it from traditional perceptron models:

#### 2.1 **Multilayer Architecture**
- The RDP is a **multilayer network**, meaning it incrementally adds layers, each corresponding to the addition of a new dimension that facilitates linear separation. This recursive structure helps in solving NLS problems.

#### 2.2 **Guaranteed Convergence**
- RDP is designed to **guarantee convergence** for any given NLS dataset. As more dimensions are added, the input space is transformed, ensuring that eventually, all classes will be separable. This makes RDP more reliable than traditional perceptrons, which can fail to converge on NLS datasets.

#### 2.3 **No Learning Parameters**
- Unlike many neural networks that rely on learning parameters such as learning rates or weights through gradient-based optimization, RDP **does not require learning parameters**. Instead, it constructs the network deterministically based on the data's linear separability properties.

#### 2.4 **No Catastrophic Forgetting**
- RDP avoids the problem of **catastrophic interference** (or forgetting) commonly seen in multilayer neural networks. As the network grows by adding new dimensions, it retains the ability to separate earlier layers' data without disrupting previously learned separations.

#### 2.5 **Transparent Knowledge Extraction**
- RDP provides a transparent method for **knowledge extraction** from the network. Each added layer corresponds to a clear and interpretable geometric transformation (a hyperplane) in the feature space.

#### 2.6 **Generalization Capabilities**
- The RDP offers generalization performance comparable to other methods like **Backpropagation (BP)**, **Cascade Correlation (CC)**, and **Rulex**, as demonstrated in its performance on benchmark datasets such as **Iris**, **Soybean**, and **Wisconsin Breast Cancer**.

---

### 3. **Construction Principle of RDP**

The RDP's construction follows a **recursive principle**, which can be broken down into the following steps:

#### 3.1 **Select Linearly Separable Subsets**
- **Identify LS Subsets**: The first step is to find **linearly separable subsets** from within the original NLS dataset. These subsets are identified based on their geometric properties (e.g., by finding vertices of the convex hull).
- **Apply a Hyperplane**: For each identified LS subset, a hyperplane is constructed using a weight vector \( w \) and bias term \( b \). The hyperplane separates this LS subset from the rest of the points.

#### 3.2 **Augment the Input Space**
- The next step is to **augment the input space** by adding a new dimension corresponding to the LS subset separation. The input vector is augmented as:

  $
  x' = [x_1, x_2, ..., x_d, h]
  $

  where $ h = w^T x + b $ represents the hyperplane for the LS subset.

#### 3.3 **Mark LS Subsets as Used**
- Once an LS subset has been successfully separated, it is marked as "used," and the algorithm moves on to the remaining NLS data points.

#### 3.4 **Recursive Addition of Layers**
- The process repeats, recursively adding dimensions until all points are separated, transforming the NLS problem into an LS one.

#### 3.5 **Termination**
- The recursion terminates when either there are no more data points left to separate or the remaining NLS problem has become LS due to the dimensional augmentation.

---

### 4. **Example Calculation: XOR Problem**

The XOR problem is a classical example of an NLS problem where traditional perceptrons fail to classify the data. We can apply the RDP construction principle to solve this problem.

#### 4.1 **XOR Dataset**
The XOR problem has the following input-output pairs:
$
\begin{aligned}
    (0,0) &\rightarrow 0 \\
    (0,1) &\rightarrow 1 \\
    (1,0) &\rightarrow 1 \\
    (1,1) &\rightarrow 0
\end{aligned}
$
This dataset is not linearly separable in a two-dimensional space because no straight line can separate the points into two distinct classes.

#### 4.2 **Step-by-Step Application of RDP**

1. **Identify LS Subsets**: 
   - The points $ (0,0) $ and $ (1,1) $ form one LS subset (labeled as class 0).
   - The points $ (0,1) $ and $ (1,0) $ form another LS subset (labeled as class 1).

2. **First Hyperplane**:
   - Construct a hyperplane that separates $ (0,0) $ and $ (1,1) $ from the rest of the points.
   - The equation of the hyperplane might look like:
     $
     h_1 = x_1 - x_2 = 0
     $
   - This separates the two subsets.

3. **Augment Input**:
   - Add the output of this separation as a new input feature. The augmented input vector becomes:
     $
     x' = [x_1, x_2, h_1]
     $
     where $ h_1 $ is the output of the first hyperplane.

4. **Second Hyperplane**:
   - Construct a second hyperplane in the augmented space to separate $ (0,1) $ and $ (1,0) $ from the others.

5. **Recursive Addition**:
   - Continue this process recursively until the XOR problem is linearly separable in the augmented space.

#### 4.3 **Final Solution**
By recursively adding dimensions to the input space, the XOR problem can be made linearly separable. In this example, the XOR problem requires at least three dimensions (two inputs and one hidden unit) for successful classification. 


### **Methods of constructing Recursive Deterministic Perceptron (RDP)**
The RDP offers several methods for constructing the neural network based on the structure and complexity of the classification problem. Each method is tailored to a specific approach to identifying linearly separable (LS) subsets within non-linearly separable (NLS) data and adding layers to the network. Below are the primary methods for building RDP neural networks:

---

### 1. **Batch Method**

#### 1.1 **Overview**
The Batch method is a **global approach** where the RDP identifies **all linearly separable subsets** (LS subsets) from the entire non-linearly separable (NLS) dataset in one go. This method looks at the complete dataset and selects LS subsets with the largest cardinality (i.e., the largest possible subset that is linearly separable).

#### 1.2 **Procedure**
1. **Selection of LS Subsets**: 
   - Identify LS subsets that belong to the same class and have **maximum cardinality**.
   - Each LS subset is separated by a hyperplane.
   
2. **Augmentation of Input Vector**:
   - Once an LS subset is separated, add an extra dimension to the input vector corresponding to the output of the separating hyperplane.
   - The augmented input vector now includes this new dimension, allowing further separation of the remaining points.

3. **Recursive Layer Addition**:
   - Repeat the process of identifying LS subsets, adding dimensions, and separating the remaining data.
   - Mark LS subsets as "used" and continue until all points in the dataset are either separated or transformed into a linearly separable (LS) problem.

#### 1.3 **Advantages**
- **Produces a smaller topology**: The Batch method creates a network with fewer intermediate neurons because the largest LS subsets are used, reducing the number of layers required.
- **Comprehensive analysis**: It takes a global view of the dataset, ensuring that the separation is as efficient as possible.

#### 1.4 **Disadvantages**
- **Computational Complexity**: The Batch method is **NP-complete**. Identifying the largest LS subsets from NLS data is computationally expensive and becomes intractable for large datasets.
- **Slower convergence**: Due to the complex nature of separating the largest subsets at once, this method can take significantly longer compared to other methods.

---

### 2. **Incremental Method**

#### 2.1 **Overview**
The Incremental method takes a **step-by-step approach** by incrementally adding one point at a time to the LS subsets. It starts by selecting an LS subset and then gradually builds the network by adding more points to it. Whenever a point is misclassified, a new hyperplane is added to separate it.

#### 2.2 **Steps in the Incremental Method**
1. **Start with a Subset**:
   - Begin with a subset of points that belong to the class with the **maximum cardinality**.
   - This initial subset is linearly separable, and a hyperplane is used to separate it from the rest.

2. **Add Points Gradually**:
   - One point is added at a time from the remaining points in the dataset.
   - If the new point is correctly classified by the existing hyperplane, no change is made.
   - If the new point is misclassified, a new hyperplane is added to separate the misclassified point from the rest of the data.

3. **Recursive Layer Addition**:
   - Continue adding points incrementally until the dataset becomes fully separable.

#### 2.3 **Advantages**
- **Lower computational complexity**: The Incremental method has **O(n log n)** complexity, making it much faster than the Batch method.
- **Dynamic adjustment**: The network evolves dynamically as each new point is added, reducing the need for recomputation of the entire dataset at every step.

#### 2.4 **Disadvantages**
- **Potential for larger topologies**: Since points are added incrementally, more intermediate neurons may be required compared to the Batch method, resulting in larger topologies.
- **Locally optimal solutions**: The Incremental method may not always find the global best solution, as it focuses on one point at a time.

---

### 3. **Modular Method**

#### 3.1 **Overview**
The Modular method is a **divide-and-conquer approach** that splits the original NLS problem into smaller, more manageable subproblems. Each subproblem is then solved independently using an RDP network (either Batch or Incremental), and the sub-solutions are combined into a larger network to solve the original problem.

#### 3.2 **Steps in the Modular Method**
1. **Divide the Original Problem**:
   - Split the original NLS dataset into smaller subproblems. Each subproblem is selected so that it is easier to solve or closer to being linearly separable.

2. **Solve Subproblems**:
   - Apply either the **Batch** or **Incremental method** to each subproblem.
   - Each subproblem results in an LS solution with its own subnetwork.

3. **Combine Subsolutions**:
   - The solutions to all subproblems are then combined to create a final RDP network that can solve the entire problem.
   - The combination is done by merging the hyperplanes of the subnetworks into a single network.

#### 3.3 **Advantages**
- **Parallelization**: The subproblems can be solved in parallel, making the Modular method **highly scalable** for large datasets.
- **Balanced complexity**: It offers a balance between the efficiency of the Incremental method and the optimality of the Batch method by focusing on smaller, manageable chunks of data.

#### 3.4 **Disadvantages**
- **Implementation complexity**: The method is more complex to implement due to the need for splitting and combining subnetworks.
- **Potential for increased network size**: The final network size may still be large if the subproblems result in many intermediate neurons.

---

### 4. **Modular-Batch and Modular-Incremental Methods**

#### 4.1 **Overview**
These hybrid methods combine the strengths of the **Modular method** with either the **Batch** or **Incremental methods** to optimize both network size and computational efficiency. 

- **Modular-Batch**: Uses the Batch method within each subproblem.
- **Modular-Incremental**: Uses the Incremental method within each subproblem.

#### 4.2 **Advantages**
- **Improved performance**: Both methods strike a balance between reducing computation time (as in the Modular method) and achieving a smaller network size (as in the Batch method).
- **Flexibility**: Depending on the dataset and problem size, the hybrid method can be adapted to use the most efficient approach for each subproblem.

#### 4.3 **Disadvantages**
- **Complexity in choosing subproblems**: Efficiently splitting the original problem into optimal subproblems can be challenging and may require problem-specific heuristics.

---

### 5. **Example of Construction**

Let's consider an example where we apply the RDP **Incremental method** to a **2-class classification problem**. The dataset includes five points that belong to two different classes:

- **Class A**: $(1, 1), (2, 2), (3, 3)$
- **Class B**: $(4, 5), (5, 6)$

#### 5.1 Steps:
1. **Start with Class A**: 
   - The points in Class A are linearly separable, so we start by separating these points using a hyperplane $h_1$.
   - $ h_1: x_1 = x_2 $.

2. **Add Class B Points Incrementally**:
   - Next, we add points from Class B. Adding point $(4, 5)$ requires us to update the separating hyperplane since it is misclassified by $ h_1 $. A new hyperplane $h_2$ is introduced to separate this point from Class A.
   - As we add the final point $(5, 6)$, it is already separated by $ h_2 $, so no further updates are required.

3. **Final Network**:
   - The final RDP network will consist of two layers (one for each hyperplane), with both layers recursively separating Class A from Class B.

---

### Conclusion

The methods for building RDP neural networks—**Batch**, **Incremental**, **Modular**, and their hybrid variations—offer flexibility in handling non-linearly separable problems. Each method has distinct advantages and disadvantages in terms of computational complexity, network size, and generalization ability, allowing practitioners to choose the most suitable approach for their specific dataset and problem.
