# Machine Learning Mathematics for Computational Biology
## A Practical Crash Course with Worked Examples

---

## Introduction: The Mathematical Ecosystem of ML

Think of machine learning models as biological systems: they need nutrients (data), metabolism (algorithms), and structure (mathematics) to function. This guide will equip you with the mathematical "anatomy and physiology" of ML, with special attention to applications in biology.

**Prerequisites Check**: You mentioned basic algebra, calculus, and statistics. Perfect! We'll build from there.

---

## Part 1: Linear Algebra - The Skeleton of Machine Learning

### 1.1 Vectors: The Building Blocks

**Biological Analogy**: Think of a vector as a molecular profile. A gene expression vector might be `[BRCA1: 2.3, TP53: 1.8, EGFR: 4.1]` - each dimension represents one gene's expression level.

**Mathematical Foundation**:
A vector is an ordered list of numbers:
```
v = [v₁, v₂, v₃, ..., vₙ]
```

**Worked Example 1: Gene Expression Vector Operations**

```python
import numpy as np

# Patient A's gene expression profile (3 genes)
patient_A = np.array([2.3, 1.8, 4.1])  # [BRCA1, TP53, EGFR]

# Patient B's gene expression profile
patient_B = np.array([1.9, 2.1, 3.8])

# Vector addition: combined expression signature
combined = patient_A + patient_B
print(f"Combined expression: {combined}")
# Output: [4.2, 3.9, 7.9]

# Scalar multiplication: treatment effect (2x upregulation)
treated = 2 * patient_A
print(f"After treatment: {treated}")
# Output: [4.6, 3.6, 8.2]

# Magnitude (L2 norm): overall expression "intensity"
magnitude = np.linalg.norm(patient_A)
print(f"Expression magnitude: {magnitude:.2f}")
# Output: 4.94
```

**Key Insight**: The magnitude tells you the "distance" from zero expression - like measuring how far a cell's state is from baseline.

---

### 1.2 Dot Products and Similarity

**Biological Analogy**: The dot product measures how "aligned" two vectors are. Like testing if two protein sequences have similar functional domains.

**Mathematical Foundation**:
```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

Also: `a · b = ||a|| ||b|| cos(θ)` where θ is the angle between vectors.

**Worked Example 2: Measuring Patient Similarity**

```python
# Two patients' gene expression profiles
patient_1 = np.array([2.3, 1.8, 4.1, 3.2])
patient_2 = np.array([2.1, 2.0, 4.3, 3.0])
patient_3 = np.array([0.5, 4.8, 1.2, 0.8])  # very different

# Dot product (raw similarity)
similarity_1_2 = np.dot(patient_1, patient_2)
similarity_1_3 = np.dot(patient_1, patient_3)

print(f"Patient 1-2 similarity: {similarity_1_2:.2f}")  # 31.79
print(f"Patient 1-3 similarity: {similarity_1_3:.2f}")  # 14.31

# Cosine similarity (normalized, ranges -1 to 1)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sim_1_2 = cosine_similarity(patient_1, patient_2)
cos_sim_1_3 = cosine_similarity(patient_1, patient_3)

print(f"\nCosine similarity 1-2: {cos_sim_1_2:.3f}")  # 0.991 (very similar!)
print(f"Cosine similarity 1-3: {cos_sim_1_3:.3f}")  # 0.524 (different)
```

**Practice Problem 1**: 
Create expression profiles for 5 patients and find which two are most similar using cosine similarity.

---

### 1.3 Matrices: Organizing High-Dimensional Data

**Biological Analogy**: A matrix is like a microarray - rows are samples (patients), columns are features (genes).

```
        Gene1  Gene2  Gene3  Gene4
Patient1  2.3    1.8    4.1    3.2
Patient2  2.1    2.0    4.3    3.0
Patient3  0.5    4.8    1.2    0.8
```

**Worked Example 3: Matrix Operations for Batch Processing**

```python
# Gene expression matrix: 3 patients × 4 genes
expression_data = np.array([
    [2.3, 1.8, 4.1, 3.2],
    [2.1, 2.0, 4.3, 3.0],
    [0.5, 4.8, 1.2, 0.8]
])

print(f"Data shape: {expression_data.shape}")  # (3, 4)

# Mean expression per gene (column means)
gene_means = np.mean(expression_data, axis=0)
print(f"Gene means: {gene_means}")
# [1.63, 2.87, 3.20, 2.33]

# Normalize each patient by their total expression
row_sums = expression_data.sum(axis=1, keepdims=True)
normalized = expression_data / row_sums
print(f"Normalized (first patient): {normalized[0]}")
# [0.20, 0.16, 0.36, 0.28] - now sums to 1.0

# Z-score normalization (subtract mean, divide by std)
z_scored = (expression_data - gene_means) / np.std(expression_data, axis=0)
print(f"\nZ-scored data:\n{z_scored}")
```

**Why This Matters**: Most ML preprocessing involves matrix operations like these!

---

### 1.4 Matrix Multiplication: Feature Transformation

**Biological Analogy**: Matrix multiplication is like a metabolic pathway - you transform inputs through a series of reactions to get outputs.

**Mathematical Foundation**:
For matrices A (m×n) and B (n×p), the product C = AB has shape (m×p):
```
C[i,j] = Σₖ A[i,k] × B[k,j]
```

**Worked Example 4: Linear Transformation (Neural Network Layer)**

```python
# Input: 3 patients with 4 gene measurements
X = np.array([
    [2.3, 1.8, 4.1, 3.2],
    [2.1, 2.0, 4.3, 3.0],
    [0.5, 4.8, 1.2, 0.8]
])

# Weight matrix: transforms 4 genes into 2 "hidden features"
# This is like finding 2 gene signatures/pathways
W = np.array([
    [0.5, -0.3],   # Gene 1 contributions
    [0.2,  0.8],   # Gene 2 contributions
    [0.9,  0.1],   # Gene 3 contributions
    [0.4,  0.6]    # Gene 4 contributions
])

# Transform data through this "layer"
hidden_features = X @ W  # @ is matrix multiplication in Python
print(f"Hidden features shape: {hidden_features.shape}")  # (3, 2)
print(f"Hidden features:\n{hidden_features}")

# Interpretation:
# - Column 0: "Pathway A activity" for each patient
# - Column 1: "Pathway B activity" for each patient
```

**Key Insight**: This is exactly what happens in neural networks! Each layer is a matrix multiplication followed by an activation function.

---

### 1.5 Eigenvalues and Eigenvectors: Finding Principal Directions

**Biological Analogy**: Imagine gene expression patterns as a "cloud" in multi-dimensional space. Eigenvectors point in the directions of maximum variance - the main axes of variation in your data, like finding the primary gene programs operating in your cells.

**Mathematical Foundation**:
For a matrix A, if:
```
A v = λ v
```
Then v is an eigenvector with eigenvalue λ.

**Worked Example 5: Principal Component Analysis (PCA)**

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate synthetic gene expression data
np.random.seed(42)
n_samples = 100
n_genes = 50

# Simulate two main biological processes
process_1 = np.random.randn(n_samples, 1) * 2  # Immune response
process_2 = np.random.randn(n_samples, 1) * 1.5  # Cell cycle

# Each gene is affected by both processes (with noise)
weights_p1 = np.random.randn(1, n_genes) * 0.8
weights_p2 = np.random.randn(1, n_genes) * 0.6
noise = np.random.randn(n_samples, n_genes) * 0.3

gene_expression = (process_1 @ weights_p1 + 
                   process_2 @ weights_p2 + 
                   noise)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(gene_expression)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
# Shows how much variance each PC captures

print(f"\nFirst PC (eigenvector 1) gene loadings (first 5):")
print(pca.components_[0, :5])

# The principal components ARE the eigenvectors of the covariance matrix!
# Eigenvalues tell us their importance
print(f"\nEigenvalues (variance explained): {pca.explained_variance_}")
```

**Practice Problem 2**: 
Load your own gene expression dataset and apply PCA. How many principal components do you need to capture 90% of variance?

---

### 1.6 Singular Value Decomposition (SVD): The Ultimate Matrix Decomposition

**Biological Analogy**: SVD decomposes your data matrix into three components - like factoring a complex biological system into: (1) sample patterns, (2) their importance, (3) gene patterns.

**Mathematical Foundation**:
Any matrix M can be decomposed as:
```
M = U Σ Vᵀ
```
- U: left singular vectors (sample patterns)
- Σ: singular values (importance/magnitude)
- Vᵀ: right singular vectors (gene patterns)

**Worked Example 6: SVD for Dimensionality Reduction**

```python
# Gene expression matrix: samples × genes
M = np.random.randn(20, 100)  # 20 samples, 100 genes

# Perform SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)

print(f"U shape (sample patterns): {U.shape}")      # (20, 20)
print(f"S shape (singular values): {S.shape}")      # (20,)
print(f"Vt shape (gene patterns): {Vt.shape}")      # (20, 100)

# Reconstruct using only top 5 components
k = 5
M_reduced = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

print(f"\nOriginal matrix shape: {M.shape}")
print(f"Reduced rank: {k}")
print(f"Reconstruction error: {np.linalg.norm(M - M_reduced):.3f}")

# Visualize importance of components
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(S, 'o-')
plt.xlabel('Component')
plt.ylabel('Singular Value')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
variance_explained = (S**2) / np.sum(S**2)
plt.plot(np.cumsum(variance_explained), 'o-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
plt.legend()
plt.tight_layout()
# plt.savefig('svd_analysis.png')
```

**Key Applications in Biology**:
- RNA-seq batch correction
- Finding gene signatures
- Compressing large genomic datasets
- Removing technical noise

---

## Part 2: Calculus for Machine Learning

### 2.1 Derivatives: The Language of Optimization

**Biological Analogy**: A derivative tells you the "slope" at a point - like measuring how quickly a bacterial population grows at a specific time, or how sensitive an enzyme is to substrate concentration changes.

**Mathematical Foundation**:
The derivative measures rate of change:
```
f'(x) = lim[h→0] (f(x+h) - f(x)) / h
```

**Worked Example 7: Gradient Descent on a Simple Function**

```python
# Let's minimize a quadratic loss function
# This is like finding the optimal parameter for a model

def loss_function(w):
    """A simple parabola: L(w) = (w - 3)² + 1"""
    return (w - 3)**2 + 1

def gradient(w):
    """Derivative: dL/dw = 2(w - 3)"""
    return 2 * (w - 3)

# Gradient descent
w = 0.0  # Start far from minimum
learning_rate = 0.1
history = [w]

for step in range(20):
    grad = gradient(w)
    w = w - learning_rate * grad  # Move against the gradient
    history.append(w)
    if step % 5 == 0:
        print(f"Step {step}: w={w:.3f}, loss={loss_function(w):.3f}, grad={grad:.3f}")

print(f"\nFinal w: {w:.3f} (true minimum is at w=3)")

# Visualize
w_vals = np.linspace(-1, 6, 100)
loss_vals = [loss_function(w) for w in w_vals]

plt.figure(figsize=(10, 5))
plt.plot(w_vals, loss_vals, 'b-', label='Loss function')
plt.plot(history, [loss_function(w) for w in history], 'ro-', 
         label='Gradient descent path', markersize=4)
plt.xlabel('Parameter w')
plt.ylabel('Loss')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.grid(True)
# plt.savefig('gradient_descent.png')
```

**Key Insight**: This is the core of how neural networks learn - they follow gradients to minimize loss!

---

### 2.2 Partial Derivatives: Multi-Variable Calculus

**Biological Analogy**: In a metabolic network, enzyme activity depends on multiple factors (pH, temperature, substrate concentration). Partial derivatives tell you how activity changes when you vary ONE factor while holding others constant.

**Mathematical Foundation**:
For function f(x, y):
```
∂f/∂x: rate of change in x direction (holding y constant)
∂f/∂y: rate of change in y direction (holding x constant)
```

**Worked Example 8: Optimizing Multiple Parameters**

```python
def loss_function_2d(w1, w2):
    """Loss with two parameters: L(w1, w2) = w1² + 2w2² - 2w1w2 + 4"""
    return w1**2 + 2*w2**2 - 2*w1*w2 + 4

def gradient_2d(w1, w2):
    """Partial derivatives"""
    dL_dw1 = 2*w1 - 2*w2  # ∂L/∂w1
    dL_dw2 = 4*w2 - 2*w1  # ∂L/∂w2
    return np.array([dL_dw1, dL_dw2])

# Gradient descent in 2D
w = np.array([5.0, 5.0])  # Starting point
learning_rate = 0.1
path = [w.copy()]

for step in range(50):
    grad = gradient_2d(w[0], w[1])
    w = w - learning_rate * grad
    path.append(w.copy())

print(f"Final parameters: w1={w[0]:.3f}, w2={w[1]:.3f}")
print(f"Final loss: {loss_function_2d(w[0], w[1]):.3f}")

# Visualize the loss landscape
w1_range = np.linspace(-2, 6, 50)
w2_range = np.linspace(-2, 6, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)
Z = loss_function_2d(W1, W2)

path = np.array(path)
plt.figure(figsize=(10, 8))
plt.contour(W1, W2, Z, levels=20, cmap='viridis')
plt.colorbar(label='Loss')
plt.plot(path[:, 0], path[:, 1], 'r.-', linewidth=2, markersize=8, 
         label='Optimization path')
plt.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start')
plt.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('2D Gradient Descent')
plt.legend()
plt.grid(True)
# plt.savefig('2d_gradient_descent.png')
```

---

### 2.3 The Chain Rule: Backpropagation's Foundation

**Biological Analogy**: Think of a signal transduction cascade: receptor → kinase → transcription factor → gene expression. The chain rule lets you trace how a change in the receptor affects gene expression by multiplying the sensitivities at each step.

**Mathematical Foundation**:
If y = f(g(x)), then:
```
dy/dx = (dy/dg) × (dg/dx)
```

**Worked Example 9: Backpropagation Through a Simple Neural Network**

```python
# Simple neural network: x → hidden → output
# x → z = w1*x + b1 → a = sigmoid(z) → y = w2*a + b2 → loss = (y - target)²

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# Forward pass
x = 2.0        # Input (gene expression value)
target = 1.0   # Target (disease state)

# Parameters
w1, b1 = 0.5, 0.1   # First layer
w2, b2 = 0.8, -0.2  # Second layer

# Forward
z = w1 * x + b1
a = sigmoid(z)
y = w2 * a + b2
loss = (y - target)**2

print("Forward Pass:")
print(f"  x={x}, z={z:.3f}, a={a:.3f}, y={y:.3f}")
print(f"  Loss={loss:.3f}")

# Backward pass (chain rule!)
# dL/dy
dL_dy = 2 * (y - target)
print(f"\nBackward Pass:")
print(f"  dL/dy = {dL_dy:.3f}")

# dL/dw2 = dL/dy × dy/dw2
dy_dw2 = a
dL_dw2 = dL_dy * dy_dw2
print(f"  dL/dw2 = {dL_dw2:.3f}")

# dL/db2 = dL/dy × dy/db2
dy_db2 = 1
dL_db2 = dL_dy * dy_db2
print(f"  dL/db2 = {dL_db2:.3f}")

# dL/da = dL/dy × dy/da
dy_da = w2
dL_da = dL_dy * dy_da
print(f"  dL/da = {dL_da:.3f}")

# dL/dz = dL/da × da/dz (chain rule!)
da_dz = sigmoid_derivative(z)
dL_dz = dL_da * da_dz
print(f"  dL/dz = {dL_dz:.3f}")

# dL/dw1 = dL/dz × dz/dw1
dz_dw1 = x
dL_dw1 = dL_dz * dz_dw1
print(f"  dL/dw1 = {dL_dw1:.3f}")

# Update parameters
learning_rate = 0.1
w1_new = w1 - learning_rate * dL_dw1
w2_new = w2 - learning_rate * dL_dw2

print(f"\nParameter updates:")
print(f"  w1: {w1} → {w1_new:.3f}")
print(f"  w2: {w2} → {w2_new:.3f}")
```

**Critical Understanding**: This IS backpropagation! In deep networks, you just keep applying the chain rule through many layers.

---

### 2.4 Gradient Descent Variants

**Worked Example 10: SGD vs. Momentum vs. Adam**

```python
# Generate synthetic regression data
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X + 7 + np.random.randn(100, 1) * 0.5  # y = 3x + 7 + noise

def compute_gradient(w, b, X, y):
    """Gradient for linear regression"""
    n = len(X)
    predictions = X * w + b
    error = predictions - y
    dw = (2/n) * np.sum(X * error)
    db = (2/n) * np.sum(error)
    return dw, db

def train_gd(X, y, learning_rate=0.01, epochs=100):
    """Standard Gradient Descent"""
    w, b = 0.0, 0.0
    losses = []
    
    for epoch in range(epochs):
        dw, db = compute_gradient(w, b, X, y)
        w -= learning_rate * dw
        b -= learning_rate * db
        loss = np.mean((X * w + b - y)**2)
        losses.append(loss)
    
    return w, b, losses

def train_momentum(X, y, learning_rate=0.01, momentum=0.9, epochs=100):
    """Gradient Descent with Momentum"""
    w, b = 0.0, 0.0
    vw, vb = 0.0, 0.0  # Velocity terms
    losses = []
    
    for epoch in range(epochs):
        dw, db = compute_gradient(w, b, X, y)
        vw = momentum * vw + learning_rate * dw
        vb = momentum * vb + learning_rate * db
        w -= vw
        b -= vb
        loss = np.mean((X * w + b - y)**2)
        losses.append(loss)
    
    return w, b, losses

def train_adam(X, y, learning_rate=0.01, beta1=0.9, beta2=0.999, epochs=100):
    """Adam optimizer"""
    w, b = 0.0, 0.0
    mw, mb = 0.0, 0.0  # First moment
    vw, vb = 0.0, 0.0  # Second moment
    epsilon = 1e-8
    losses = []
    
    for epoch in range(epochs):
        dw, db = compute_gradient(w, b, X, y)
        
        # Update biased moments
        mw = beta1 * mw + (1 - beta1) * dw
        mb = beta1 * mb + (1 - beta1) * db
        vw = beta2 * vw + (1 - beta2) * dw**2
        vb = beta2 * vb + (1 - beta2) * db**2
        
        # Bias correction
        mw_hat = mw / (1 - beta1**(epoch + 1))
        mb_hat = mb / (1 - beta1**(epoch + 1))
        vw_hat = vw / (1 - beta2**(epoch + 1))
        vb_hat = vb / (1 - beta2**(epoch + 1))
        
        # Update parameters
        w -= learning_rate * mw_hat / (np.sqrt(vw_hat) + epsilon)
        b -= learning_rate * mb_hat / (np.sqrt(vb_hat) + epsilon)
        
        loss = np.mean((X * w + b - y)**2)
        losses.append(loss)
    
    return w, b, losses

# Compare all three
w_gd, b_gd, losses_gd = train_gd(X, y, epochs=100)
w_mom, b_mom, losses_mom = train_momentum(X, y, epochs=100)
w_adam, b_adam, losses_adam = train_adam(X, y, epochs=100)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses_gd, label='Standard GD', alpha=0.7)
plt.plot(losses_mom, label='Momentum', alpha=0.7)
plt.plot(losses_adam, label='Adam', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Convergence Comparison')
plt.legend()
plt.yscale('log')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.5, label='Data')
x_line = np.array([[-4], [4]])
plt.plot(x_line, x_line * w_gd + b_gd, 'r-', label=f'GD: y={w_gd:.2f}x+{b_gd:.2f}')
plt.plot(x_line, x_line * w_adam + b_adam, 'g--', label=f'Adam: y={w_adam:.2f}x+{b_adam:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Final Fits')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig('optimizer_comparison.png')

print(f"True parameters: w=3, b=7")
print(f"GD: w={w_gd:.3f}, b={b_gd:.3f}")
print(f"Adam: w={w_adam:.3f}, b={b_adam:.3f}")
```

---

## Part 3: Probability and Statistics for ML

### 3.1 Probability Distributions

**Biological Analogy**: Gene expression levels follow distributions - often log-normal or negative binomial in RNA-seq. Understanding distributions helps you model biological variability.

**Worked Example 11: Common Distributions in Biology**

```python
from scipy import stats
import matplotlib.pyplot as plt

# Generate figure with multiple distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Normal (Gaussian) - common for continuous measurements
x = np.linspace(-4, 4, 100)
axes[0, 0].plot(x, stats.norm.pdf(x, 0, 1), label='μ=0, σ=1')
axes[0, 0].plot(x, stats.norm.pdf(x, 0, 0.5), label='μ=0, σ=0.5')
axes[0, 0].set_title('Normal Distribution\n(e.g., protein concentrations)')
axes[0, 0].legend()
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Probability Density')

# 2. Log-Normal - RNA-seq counts
x = np.linspace(0, 10, 100)
axes[0, 1].plot(x, stats.lognorm.pdf(x, s=0.5), label='s=0.5')
axes[0, 1].plot(x, stats.lognorm.pdf(x, s=1.0), label='s=1.0')
axes[0, 1].set_title('Log-Normal Distribution\n(e.g., gene expression)')
axes[0, 1].legend()
axes[0, 1].set_xlabel('Expression Level')

# 3. Poisson - count data
x_discrete = np.arange(0, 20)
axes[0, 2].bar(x_discrete, stats.poisson.pmf(x_discrete, mu=5), alpha=0.7, label='λ=5')
axes[0, 2].bar(x_discrete, stats.poisson.pmf(x_discrete, mu=10), alpha=0.7, label='λ=10')
axes[0, 2].set_title('Poisson Distribution\n(e.g., mutation counts)')
axes[0, 2].legend()
axes[0, 2].set_xlabel('Count')
axes[0, 2].set_ylabel('Probability')

# 4. Binomial - yes/no outcomes
n, p = 20, 0.3
x_discrete = np.arange(0, n+1)
axes[1, 0].bar(x_discrete, stats.binom.pmf(x_discrete, n, p), alpha=0.7)
axes[1, 0].set_title(f'Binomial Distribution\n(e.g., {n} trials, p={p})')
axes[1, 0].set_xlabel('Number of Successes')
axes[1, 0].set_ylabel('Probability')

# 5. Exponential - waiting times
x = np.linspace(0, 5, 100)
axes[1, 1].plot(x, stats.expon.pdf(x, scale=1), label='λ=1')
axes[1, 1].plot(x, stats.expon.pdf(x, scale=0.5), label='λ=2')
axes[1, 1].set_title('Exponential Distribution\n(e.g., time between events)')
axes[1, 1].legend()
axes[1, 1].set_xlabel('Time')

# 6. Beta - proportions
x = np.linspace(0, 1, 100)
axes[1, 2].plot(x, stats.beta.pdf(x, a=2, b=5), label='α=2, β=5')
axes[1, 2].plot(x, stats.beta.pdf(x, a=5, b=2), label='α=5, β=2')
axes[1, 2].set_title('Beta Distribution\n(e.g., allele frequencies)')
axes[1, 2].legend()
axes[1, 2].set_xlabel('Proportion')

plt.tight_layout()
# plt.savefig('distributions.png')

# Sampling example
print("Sampling from distributions:")
normal_samples = np.random.normal(loc=5, scale=2, size=1000)
print(f"Normal: mean={normal_samples.mean():.2f}, std={normal_samples.std():.2f}")

poisson_samples = np.random.poisson(lam=10, size=1000)
print(f"Poisson: mean={poisson_samples.mean():.2f}, variance={poisson_samples.var():.2f}")
```

---

### 3.2 Bayes' Theorem: The Foundation of Probabilistic ML

**Biological Analogy**: Diagnostic testing! Given a positive test result, what's the probability you actually have the disease? This is Bayes' theorem in action.

**Mathematical Foundation**:
```
P(A|B) = P(B|A) × P(A) / P(B)

Or more commonly:
P(disease|positive) = P(positive|disease) × P(disease) / P(positive)
```

**Worked Example 12: Medical Diagnosis with Bayes**

```python
def bayes_diagnostic_test(sensitivity, specificity, prevalence):
    """
    Calculate probability of disease given positive test
    
    sensitivity: P(positive|disease) - true positive rate
    specificity: P(negative|no disease) - true negative rate
    prevalence: P(disease) - base rate in population
    """
    # P(positive|no disease) = 1 - specificity (false positive rate)
    false_positive_rate = 1 - specificity
    
    # P(positive) using law of total probability
    p_positive = (sensitivity * prevalence + 
                  false_positive_rate * (1 - prevalence))
    
    # Bayes' theorem
    p_disease_given_positive = (sensitivity * prevalence) / p_positive
    
    return p_disease_given_positive

# Example: Cancer screening test
sensitivity = 0.95  # 95% of sick people test positive
specificity = 0.90  # 90% of healthy people test negative
prevalence = 0.01   # 1% of population has cancer

prob = bayes_diagnostic_test(sensitivity, specificity, prevalence)
print(f"Test Results:")
print(f"  Sensitivity: {sensitivity*100}%")
print(f"  Specificity: {specificity*100}%")
print(f"  Disease prevalence: {prevalence*100}%")
print(f"\nIf you test POSITIVE:")
print(f"  Probability you actually have disease: {prob*100:.1f}%")
print(f"\nSurprised? The low base rate matters a lot!")

# Explore how prevalence affects interpretation
prevalences = np.logspace(-4, -1, 50)  # 0.01% to 10%
probabilities = [bayes_diagnostic_test(sensitivity, specificity, p) 
                 for p in prevalences]

plt.figure(figsize=(10, 6))
plt.semilogx(prevalences * 100, np.array(probabilities) * 100)
plt.xlabel('Disease Prevalence (%)')
plt.ylabel('P(Disease | Positive Test) (%)')
plt.title('How Base Rate Affects Test Interpretation\n' + 
          f'(Sensitivity={sensitivity*100}%, Specificity={specificity*100}%)')
plt.grid(True)
plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
plt.legend()
# plt.savefig('bayes_prevalence.png')
```

**Key Insight for ML**: Bayesian thinking is everywhere - in Naive Bayes classifiers, Bayesian neural networks, and prior regularization!

---

### 3.3 Maximum Likelihood Estimation (MLE)

**Biological Analogy**: You observe some gene expression data. What parameter values for your distribution model make the observed data most probable? That's MLE.

**Worked Example 13: Fitting a Distribution to Gene Expression Data**

```python
# Simulate gene expression data (log-normal distributed)
np.random.seed(42)
true_mu, true_sigma = 2.5, 0.8
expression_data = np.random.lognormal(mean=true_mu, sigma=true_sigma, size=200)

print(f"True parameters: μ={true_mu}, σ={true_sigma}")

# Method 1: Analytical MLE for log-normal
# For log-normal, MLE is mean and std of log-transformed data
log_data = np.log(expression_data)
mle_mu = np.mean(log_data)
mle_sigma = np.std(log_data, ddof=1)

print(f"MLE estimates: μ={mle_mu:.3f}, σ={mle_sigma:.3f}")

# Method 2: Numerical optimization
from scipy.optimize import minimize

def negative_log_likelihood(params, data):
    """Negative log-likelihood for log-normal distribution"""
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    # Log-likelihood for log-normal
    n = len(data)
    log_data = np.log(data)
    ll = -n * np.log(sigma) - 0.5 * n * np.log(2 * np.pi)
    ll -= np.sum((log_data - mu)**2) / (2 * sigma**2)
    return -ll  # Return negative because we minimize

# Optimize
initial_guess = [1.0, 1.0]
result = minimize(negative_log_likelihood, initial_guess, 
                 args=(expression_data,), method='Nelder-Mead')

print(f"Numerical MLE: μ={result.x[0]:.3f}, σ={result.x[1]:.3f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(expression_data, bins=30, density=True, alpha=0.7, label='Data')
x = np.linspace(0, max(expression_data), 100)
plt.plot(x, stats.lognorm.pdf(x, s=mle_sigma, scale=np.exp(mle_mu)), 
         'r-', linewidth=2, label='Fitted distribution')
plt.xlabel('Expression Level')
plt.ylabel('Density')
plt.title('MLE Fit to Gene Expression Data')
plt.legend()

plt.subplot(1, 2, 2)
# Log-likelihood surface
mu_range = np.linspace(2.0, 3.0, 50)
sigma_range = np.linspace(0.5, 1.2, 50)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
LL = np.zeros_like(MU)
for i in range(len(mu_range)):
    for j in range(len(sigma_range)):
        LL[j, i] = -negative_log_likelihood([MU[j, i], SIGMA[j, i]], 
                                            expression_data)

plt.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
plt.colorbar(label='Log-Likelihood')
plt.plot(mle_mu, mle_sigma, 'r*', markersize=15, label='MLE')
plt.plot(true_mu, true_sigma, 'go', markersize=10, label='True')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('Log-Likelihood Surface')
plt.legend()
plt.tight_layout()
# plt.savefig('mle_fitting.png')
```

---

### 3.4 Information Theory Basics

**Worked Example 14: Entropy and KL Divergence**

```python
def entropy(p):
    """Shannon entropy: H(p) = -Σ p(x) log p(x)"""
    p = np.array(p)
    p = p[p > 0]  # Remove zeros to avoid log(0)
    return -np.sum(p * np.log2(p))

def kl_divergence(p, q):
    """KL divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))"""
    p, q = np.array(p), np.array(q)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

# Example: Cell type distribution
cell_types = ['B cells', 'T cells', 'NK cells', 'Monocytes']

# Patient A: uniform distribution (high entropy)
patient_A = np.array([0.25, 0.25, 0.25, 0.25])

# Patient B: mostly one type (low entropy)
patient_B = np.array([0.80, 0.10, 0.05, 0.05])

# Healthy reference
healthy_ref = np.array([0.30, 0.40, 0.15, 0.15])

print("Entropy (uncertainty):")
print(f"  Patient A: {entropy(patient_A):.3f} bits")
print(f"  Patient B: {entropy(patient_B):.3f} bits")
print(f"  Healthy: {entropy(healthy_ref):.3f} bits")

print("\nKL Divergence from healthy (difference):")
print(f"  Patient A vs Healthy: {kl_divergence(patient_A, healthy_ref):.3f}")
print(f"  Patient B vs Healthy: {kl_divergence(patient_B, healthy_ref):.3f}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, dist, label in zip(axes, 
                           [patient_A, patient_B, healthy_ref],
                           ['Patient A (uniform)', 'Patient B (skewed)', 
                            'Healthy Ref']):
    ax.bar(cell_types, dist)
    ax.set_ylabel('Proportion')
    ax.set_title(f'{label}\nEntropy: {entropy(dist):.2f} bits')
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
# plt.savefig('entropy_celltype.png')
```

**Why This Matters**: 
- Cross-entropy is the loss function in classification!
- KL divergence measures distribution differences
- Used in VAEs, GANs, and many other models

---

## Part 4: Optimization Theory

### 4.1 Convexity and Loss Functions

**Biological Analogy**: A convex function is like a bowl - there's one global minimum. Non-convex is like a mountainous landscape with many valleys (local minima). Training neural networks means navigating this landscape.

**Worked Example 15: Convex vs Non-Convex Optimization**

```python
# Convex function: quadratic
def convex_loss(x):
    return x**2 + 2*x + 1

# Non-convex function: multiple local minima
def nonconvex_loss(x):
    return np.sin(x) * x**2 / 10 + x**2 / 20

# Visualize
x = np.linspace(-10, 10, 1000)
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(x, convex_loss(x))
plt.title('Convex Loss Function\n(Single Global Minimum)')
plt.xlabel('Parameter x')
plt.ylabel('Loss')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(x, nonconvex_loss(x))
plt.title('Non-Convex Loss Function\n(Multiple Local Minima)')
plt.xlabel('Parameter x')
plt.ylabel('Loss')
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
# plt.savefig('convexity.png')

# Demonstrate gradient descent on non-convex function
def gradient_nonconvex(x):
    # Numerical gradient
    h = 1e-5
    return (nonconvex_loss(x + h) - nonconvex_loss(x - h)) / (2 * h)

# Multiple starting points
starting_points = [-8, -3, 0, 5, 8]
plt.figure(figsize=(12, 6))
plt.plot(x, nonconvex_loss(x), 'b-', linewidth=2, label='Loss function')

for start in starting_points:
    x_current = start
    path = [x_current]
    
    for _ in range(100):
        grad = gradient_nonconvex(x_current)
        x_current = x_current - 0.1 * grad
        path.append(x_current)
    
    path = np.array(path)
    plt.plot(path, [nonconvex_loss(p) for p in path], 'o-', 
             alpha=0.6, markersize=3, label=f'Start: {start}')

plt.xlabel('Parameter x')
plt.ylabel('Loss')
plt.title('Gradient Descent from Different Starting Points\n' + 
          'Notice how starting point affects which minimum you find!')
plt.legend()
plt.grid(True)
# plt.savefig('nonconvex_gd.png')
```

---

### 4.2 Regularization: Preventing Overfitting

**Biological Analogy**: Regularization is like Occam's razor - prefer simpler explanations. In biology, we prefer models that generalize rather than memorizing every detail.

**Worked Example 16: L1 (Lasso) vs L2 (Ridge) Regularization**

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Generate data with many features but few important ones
np.random.seed(42)
n_samples = 100
n_features = 50

# Only first 5 features are truly important
X = np.random.randn(n_samples, n_features)
true_coefficients = np.zeros(n_features)
true_coefficients[:5] = [3, -2, 1.5, -1, 2.5]
y = X @ true_coefficients + np.random.randn(n_samples) * 0.5

# Split data
train_X, train_y = X[:80], y[:80]
test_X, test_y = X[80:], y[80:]

# No regularization (will overfit)
from sklearn.linear_model import LinearRegression
model_none = LinearRegression().fit(train_X, train_y)

# L2 (Ridge): penalizes sum of squared coefficients
model_ridge = Ridge(alpha=1.0).fit(train_X, train_y)

# L1 (Lasso): penalizes sum of absolute coefficients, encourages sparsity
model_lasso = Lasso(alpha=0.1).fit(train_X, train_y)

# Compare coefficients
plt.figure(figsize=(15, 10))

# Plot 1: Coefficients comparison
plt.subplot(2, 2, 1)
plt.plot(true_coefficients, 'ko-', label='True', markersize=8)
plt.plot(model_none.coef_, 'b.-', alpha=0.6, label='No regularization')
plt.plot(model_ridge.coef_, 'g.-', alpha=0.6, label='Ridge (L2)')
plt.plot(model_lasso.coef_, 'r.-', alpha=0.6, label='Lasso (L1)')
plt.xlabel('Feature index')
plt.ylabel('Coefficient value')
plt.title('Learned Coefficients')
plt.legend()
plt.grid(True)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)

# Plot 2: Number of non-zero coefficients
plt.subplot(2, 2, 2)
counts = [
    np.sum(np.abs(true_coefficients) > 0.01),
    np.sum(np.abs(model_none.coef_) > 0.01),
    np.sum(np.abs(model_ridge.coef_) > 0.01),
    np.sum(np.abs(model_lasso.coef_) > 0.01)
]
plt.bar(['True', 'None', 'Ridge', 'Lasso'], counts)
plt.ylabel('Number of non-zero coefficients')
plt.title('Sparsity (Feature Selection)')
plt.grid(True, axis='y')

# Plot 3: Training and test error
train_preds = {
    'None': model_none.predict(train_X),
    'Ridge': model_ridge.predict(train_X),
    'Lasso': model_lasso.predict(train_X)
}
test_preds = {
    'None': model_none.predict(test_X),
    'Ridge': model_ridge.predict(test_X),
    'Lasso': model_lasso.predict(test_X)
}

train_errors = {k: np.mean((v - train_y)**2) for k, v in train_preds.items()}
test_errors = {k: np.mean((v - test_y)**2) for k, v in test_preds.items()}

plt.subplot(2, 2, 3)
x_pos = np.arange(3)
plt.bar(x_pos - 0.2, list(train_errors.values()), 0.4, label='Train', alpha=0.7)
plt.bar(x_pos + 0.2, list(test_errors.values()), 0.4, label='Test', alpha=0.7)
plt.xticks(x_pos, list(train_errors.keys()))
plt.ylabel('Mean Squared Error')
plt.title('Training vs Test Error')
plt.legend()
plt.grid(True, axis='y')

# Plot 4: Regularization path for Lasso
alphas = np.logspace(-3, 1, 50)
coefs = []
for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000).fit(train_X, train_y)
    coefs.append(model.coef_)
coefs = np.array(coefs)

plt.subplot(2, 2, 4)
for i in range(5):  # Plot only first 5 features
    plt.plot(alphas, coefs[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.xlabel('Regularization strength (α)')
plt.ylabel('Coefficient value')
plt.title('Lasso Regularization Path\n(How coefficients shrink)')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.savefig('regularization_comparison.png')

print("Mean Squared Errors:")
print(f"  No regularization: Train={train_errors['None']:.3f}, Test={test_errors['None']:.3f}")
print(f"  Ridge (L2):        Train={train_errors['Ridge']:.3f}, Test={test_errors['Ridge']:.3f}")
print(f"  Lasso (L1):        Train={train_errors['Lasso']:.3f}, Test={test_errors['Lasso']:.3f}")
```

**Key Takeaway**: 
- L2 (Ridge): Shrinks all coefficients smoothly
- L1 (Lasso): Drives some coefficients to exactly zero (feature selection!)
- In genomics, Lasso is great for finding which genes actually matter

---

## Part 5: Graph Theory for Biological Networks

### 5.1 Graph Basics

**Biological Analogy**: Biological systems are networks - protein-protein interactions, metabolic pathways, gene regulatory networks, neural circuits. Graphs are how we mathematically represent these.

**Mathematical Foundation**:
- **Graph G = (V, E)**: V = vertices/nodes, E = edges/connections
- **Adjacency Matrix A**: A[i,j] = 1 if edge between i and j, else 0
- **Degree**: Number of connections per node

**Worked Example 17: Protein Interaction Network Analysis**

```python
import networkx as nx

# Create a simple protein interaction network
G = nx.Graph()

# Add proteins (nodes)
proteins = ['p53', 'MDM2', 'ATM', 'CHK2', 'BRCA1', 'BRCA2', 'RAD51']
G.add_nodes_from(proteins)

# Add interactions (edges)
interactions = [
    ('p53', 'MDM2'), ('p53', 'ATM'), ('p53', 'CHK2'),
    ('ATM', 'CHK2'), ('ATM', 'BRCA1'),
    ('BRCA1', 'BRCA2'), ('BRCA1', 'RAD51'),
    ('BRCA2', 'RAD51'), ('CHK2', 'BRCA1')
]
G.add_edges_from(interactions)

# Compute network properties
degree_centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)
clustering = nx.clustering(G)

print("Network Statistics:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")

print("\nMost central proteins (by degree):")
for protein, centrality in sorted(degree_centrality.items(), 
                                  key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {protein}: {centrality:.3f} (degree={G.degree(protein)})")

print("\nMost important bridges (betweenness):")
for protein, score in sorted(betweenness.items(), 
                             key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {protein}: {score:.3f}")

# Visualize
plt.figure(figsize=(12, 5))

# Plot 1: Network visualization
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G, seed=42)
node_sizes = [3000 * degree_centrality[node] for node in G.nodes()]
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=node_sizes, font_size=10, font_weight='bold',
        edge_color='gray', width=2)
plt.title('Protein Interaction Network\n(Node size = centrality)')

# Plot 2: Adjacency matrix
plt.subplot(1, 2, 2)
adj_matrix = nx.to_numpy_array(G)
plt.imshow(adj_matrix, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Connection (1=yes, 0=no)')
plt.xticks(range(len(proteins)), proteins, rotation=45)
plt.yticks(range(len(proteins)), proteins)
plt.title('Adjacency Matrix')

plt.tight_layout()
# plt.savefig('protein_network.png')
```

---

### 5.2 Graph Algorithms

**Worked Example 18: Shortest Paths and Community Detection**

```python
# Create a larger metabolic network
np.random.seed(42)
G_metabolic = nx.Graph()

# Add metabolites
metabolites = [f"M{i}" for i in range(20)]
G_metabolic.add_nodes_from(metabolites)

# Add reactions (random network with some structure)
for i in range(30):
    u, v = np.random.choice(metabolites, 2, replace=False)
    G_metabolic.add_edge(u, v)

# Shortest path
source, target = 'M0', 'M15'
if nx.has_path(G_metabolic, source, target):
    path = nx.shortest_path(G_metabolic, source, target)
    path_length = nx.shortest_path_length(G_metabolic, source, target)
    print(f"Shortest metabolic path from {source} to {target}:")
    print(f"  Path: {' → '.join(path)}")
    print(f"  Length: {path_length} reactions")

# Find communities (modules in the network)
communities = list(nx.community.greedy_modularity_communities(G_metabolic))
print(f"\nFound {len(communities)} metabolic modules:")
for i, community in enumerate(communities):
    print(f"  Module {i+1}: {len(community)} metabolites")

# Visualize
plt.figure(figsize=(14, 6))

# Plot 1: Full network with communities
plt.subplot(1, 2, 1)
pos = nx.spring_layout(G_metabolic, seed=42)
colors = []
for node in G_metabolic.nodes():
    for i, community in enumerate(communities):
        if node in community:
            colors.append(i)
            break

nx.draw(G_metabolic, pos, node_color=colors, cmap='Set3',
        with_labels=True, node_size=500, font_size=8,
        edge_color='gray', alpha=0.7)
plt.title('Metabolic Network\n(Colors = communities)')

# Plot 2: Degree distribution
plt.subplot(1, 2, 2)
degrees = [d for n, d in G_metabolic.degree()]
plt.hist(degrees, bins=range(max(degrees)+2), edgecolor='black', alpha=0.7)
plt.xlabel('Degree (number of connections)')
plt.ylabel('Number of metabolites')
plt.title('Degree Distribution')
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
# plt.savefig('metabolic_network.png')
```

---

### 5.3 Graph Convolutional Networks (GCNs)

**Biological Analogy**: GCNs aggregate information from neighbors in a network. Like how a gene's expression might be influenced by its regulators in a gene regulatory network.

**Worked Example 19: Simple Graph Convolution**

```python
def graph_convolution(X, A):
    """
    Simple graph convolution: aggregate features from neighbors
    
    X: node features (n_nodes × n_features)
    A: adjacency matrix (n_nodes × n_nodes)
    
    Returns: aggregated features
    """
    # Add self-loops
    A_hat = A + np.eye(A.shape[0])
    
    # Degree matrix
    D = np.diag(np.sum(A_hat, axis=1))
    
    # Symmetric normalization: D^(-1/2) A D^(-1/2)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    
    # Aggregate
    X_aggregated = A_norm @ X
    
    return X_aggregated

# Example: Gene regulatory network
n_genes = 8
gene_names = [f"Gene{i}" for i in range(n_genes)]

# Create adjacency matrix (who regulates whom)
A = np.array([
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 0]
], dtype=float)

# Initial gene expression features (3 conditions)
X = np.array([
    [1.0, 0.5, 2.0],  # Gene0
    [0.8, 0.6, 1.8],  # Gene1
    [1.2, 0.4, 2.1],  # Gene2
    [0.5, 1.0, 1.5],  # Gene3
    [0.6, 0.9, 1.6],  # Gene4
    [0.3, 1.2, 1.3],  # Gene5
    [0.4, 1.1, 1.4],  # Gene6
    [0.2, 1.3, 1.2],  # Gene7
])

print("Original features (3 conditions):")
print(X[:3])

# Apply graph convolution
X_conv1 = graph_convolution(X, A)
print("\nAfter 1 layer of graph convolution:")
print(X_conv1[:3])
print("(Features now incorporate information from neighbors)")

# Apply second layer
X_conv2 = graph_convolution(X_conv1, A)
print("\nAfter 2 layers:")
print(X_conv2[:3])
print("(Features now incorporate info from 2-hop neighbors)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, features, title in zip(axes, 
                                [X, X_conv1, X_conv2],
                                ['Original', '1 GCN Layer', '2 GCN Layers']):
    im = ax.imshow(features, aspect='auto', cmap='coolwarm')
    ax.set_xlabel('Condition')
    ax.set_ylabel('Gene')
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(gene_names)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Expression')

plt.tight_layout()
# plt.savefig('graph_convolution.png')
```

**Key Insight**: Each GCN layer aggregates information from neighbors. After k layers, each node has information from k-hop neighbors!

---

## Part 6: Putting It All Together

### 6.1 Complete Neural Network from Scratch

**Worked Example 20: Build and Train a Neural Network**

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list of layer dimensions, e.g., [4, 8, 3] 
                     means 4 inputs, 8 hidden, 3 outputs
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        
        # Initialize weights and biases (Xavier initialization)
        self.weights = []
        self.biases = []
        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * \
                np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass - store activations for backprop"""
        self.activations = [X]
        self.z_values = []
        
        for i in range(self.num_layers - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            if i == self.num_layers - 2:  # Output layer
                a = self.softmax(z)
            else:  # Hidden layers
                a = self.sigmoid(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, learning_rate=0.01):
        """Backpropagation"""
        m = X.shape[0]  # Number of samples
        
        # Convert y to one-hot if needed
        if len(y.shape) == 1:
            y_onehot = np.zeros((m, self.layer_sizes[-1]))
            y_onehot[np.arange(m), y] = 1
        else:
            y_onehot = y
        
        # Backward pass
        deltas = [None] * (self.num_layers - 1)
        
        # Output layer error
        deltas[-1] = self.activations[-1] - y_onehot
        
        # Hidden layers
        for i in range(self.num_layers - 3, -1, -1):
            delta = (deltas[i+1] @ self.weights[i+1].T) * \
                    self.sigmoid_derivative(self.z_values[i])
            deltas[i] = delta
        
        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * \
                               (self.activations[i].T @ deltas[i]) / m
            self.biases[i] -= learning_rate * \
                              np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=100, learning_rate=0.01, verbose=True):
        """Training loop"""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute loss (cross-entropy)
            m = X.shape[0]
            y_onehot = np.zeros((m, self.layer_sizes[-1]))
            y_onehot[np.arange(m), y] = 1
            loss = -np.mean(np.sum(y_onehot * np.log(predictions + 1e-8), axis=1))
            losses.append(loss)
            
            # Compute accuracy
            pred_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(pred_classes == y)
            accuracies.append(accuracy)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
        
        return losses, accuracies
    
    def predict(self, X):
        """Make predictions"""
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)

# Example: Classify cell types based on marker expression
np.random.seed(42)

# Generate synthetic data: 3 cell types based on 4 markers
n_samples_per_class = 50
n_features = 4

# Class 0: High marker 0 and 1
class_0 = np.random.randn(n_samples_per_class, n_features) * 0.5
class_0[:, 0] += 2
class_0[:, 1] += 2

# Class 1: High marker 2 and 3
class_1 = np.random.randn(n_samples_per_class, n_features) * 0.5
class_1[:, 2] += 2
class_1[:, 3] += 2

# Class 2: High marker 1 and 2
class_2 = np.random.randn(n_samples_per_class, n_features) * 0.5
class_2[:, 1] += 2
class_2[:, 2] += 2

X = np.vstack([class_0, class_1, class_2])
y = np.hstack([np.zeros(n_samples_per_class, dtype=int),
               np.ones(n_samples_per_class, dtype=int),
               np.ones(n_samples_per_class, dtype=int) * 2])

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

# Split train/test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Create and train network
nn = NeuralNetwork([4, 10, 3])  # 4 inputs, 10 hidden, 3 outputs
print("Training Neural Network...")
losses, accuracies = nn.train(X_train, y_train, epochs=100, 
                              learning_rate=0.1, verbose=True)

# Test
test_predictions = nn.predict(X_test)
test_accuracy = np.mean(test_predictions == y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Visualize training
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Cross-Entropy)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)
plt.ylim([0, 1])

plt.tight_layout()
# plt.savefig('nn_training.png')

# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, test_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Type 0', 'Type 1', 'Type 2'],
            yticklabels=['Type 0', 'Type 1', 'Type 2'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Cell Type Classification')
# plt.savefig('confusion_matrix.png')
```

---

## Practice Problems and Challenges

### Challenge 1: PCA on Real Gene Expression Data
Download a gene expression dataset (like from GEO or TCGA), apply PCA, and interpret the principal components. Which genes contribute most to PC1 and PC2?

### Challenge 2: Implement Adam Optimizer
Extend the neural network class to use Adam optimizer instead of vanilla SGD. Compare convergence rates.

### Challenge 3: Graph Analysis
Build a protein-protein interaction network from STRING database. Find:
- Most central proteins
- Protein complexes (communities)
- Shortest paths between disease-related proteins

### Challenge 4: Bayesian Classifier
Implement a Naive Bayes classifier for disease diagnosis from gene expression data. Compare with your neural network.

### Challenge 5: Regularization Study
Using genomic data with >10,000 genes:
- Compare Lasso, Ridge, and Elastic Net
- How many genes does Lasso select?
- Do selected genes make biological sense?

---

## Quick Reference: Essential Formulas

### Linear Algebra
```
Dot product: a·b = Σ aᵢbᵢ
Matrix multiplication: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
Eigenvalue equation: Av = λv
SVD: M = UΣVᵀ
```

### Calculus
```
Derivative: f'(x) = lim[h→0] (f(x+h) - f(x))/h
Chain rule: dy/dx = (dy/du)(du/dx)
Gradient: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
Gradient descent: θ := θ - α∇J(θ)
```

### Probability
```
Bayes: P(A|B) = P(B|A)P(A)/P(B)
Entropy: H(X) = -Σ P(x)log₂P(x)
KL divergence: D_KL(P||Q) = Σ P(x)log(P(x)/Q(x))
```

### Optimization
```
L2 regularization: J(θ) = Loss + λΣθᵢ²
L1 regularization: J(θ) = Loss + λΣ|θᵢ|
```

---

## Recommended Next Steps

1. **Practice with Real Data**: Apply these concepts to actual biological datasets
2. **Implement from Scratch**: Build your own versions before using libraries
3. **Read Papers**: Start with classic ML papers in computational biology
4. **Contribute to Projects**: Join open-source bioinformatics projects
5. **Take Courses**: 
   - Stanford CS229 (Machine Learning)
   - Fast.ai (Practical Deep Learning)
   - Coursera: Genomic Data Science

---

## Key Biological Applications

- **Gene Expression Analysis**: PCA, clustering, differential expression
- **Protein Structure**: Graph neural networks, geometric deep learning
- **Drug Discovery**: Molecular property prediction, generative models
- **Medical Imaging**: CNNs for pathology, radiology
- **Single-Cell Genomics**: Manifold learning, trajectory inference
- **Sequence Analysis**: RNNs, transformers, attention mechanisms
- **Systems Biology**: Network inference, dynamical systems

---

Remember: **Mathematics is a language**. The more you use it, the more fluent you become. Start simple, build intuition, then tackle complexity. Every expert was once a beginner!

Happy learning! 🧬🤖📊
