# ML MATH QUICK REFERENCE GUIDE
## Essential Formulas & Concepts

---

## LINEAR ALGEBRA

### Vectors
- **Dot Product**: a·b = Σ aᵢbᵢ = |a||b|cos(θ)
- **L2 Norm**: ||v|| = √(Σ vᵢ²)
- **L1 Norm**: ||v||₁ = Σ |vᵢ|
- **Cosine Similarity**: cos(θ) = (a·b)/(||a|| ||b||)
- **Euclidean Distance**: d(a,b) = ||a - b||

### Matrices
- **Matrix Multiplication**: (AB)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
  - Dimensions: (m×n)(n×p) = (m×p)
- **Transpose**: (Aᵀ)ᵢⱼ = Aⱼᵢ
- **Trace**: tr(A) = Σᵢ Aᵢᵢ
- **Determinant**: det(A) - measures volume scaling
- **Inverse**: AA⁻¹ = I (only for square, non-singular matrices)

### Matrix Properties
- (AB)ᵀ = BᵀAᵀ
- (AB)⁻¹ = B⁻¹A⁻¹
- tr(AB) = tr(BA)
- det(AB) = det(A)det(B)

### Eigendecomposition
- **Definition**: Av = λv
  - v: eigenvector (direction)
  - λ: eigenvalue (scaling)
- **Decomposition**: A = VΛVᵀ (for symmetric A)
  - V: eigenvectors (columns)
  - Λ: diagonal matrix of eigenvalues

### SVD (Singular Value Decomposition)
- **Formula**: A = UΣVᵀ
  - U (m×m): left singular vectors
  - Σ (m×n): singular values (diagonal)
  - V (n×n): right singular vectors
- **Rank-k Approximation**: Aₖ = Σᵢ₌₁ᵏ σᵢ uᵢvᵢᵀ
- **Connection to Eigendecomposition**:
  - AᵀA = VΣ²Vᵀ (right singular vectors = eigenvectors of AᵀA)
  - AAᵀ = UΣ²Uᵀ (left singular vectors = eigenvectors of AAᵀ)

### PCA (Principal Component Analysis)
1. Center data: X̃ = X - mean(X)
2. Compute covariance: C = (X̃ᵀX̃)/(n-1)
3. Eigendecomposition: C = VΛVᵀ
4. Project: Z = X̃V
5. Variance explained: λᵢ/Σλⱼ

**Python**:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
Z = pca.fit_transform(X)
variance_ratio = pca.explained_variance_ratio_
```

---

## CALCULUS

### Derivatives (Single Variable)
- **Power Rule**: d/dx(xⁿ) = nxⁿ⁻¹
- **Exponential**: d/dx(eˣ) = eˣ
- **Logarithm**: d/dx(ln x) = 1/x
- **Product Rule**: (fg)' = f'g + fg'
- **Chain Rule**: d/dx f(g(x)) = f'(g(x))·g'(x)

### Common Derivatives
- d/dx(sin x) = cos x
- d/dx(cos x) = -sin x
- d/dx(1/(1+e⁻ˣ)) = e⁻ˣ/(1+e⁻ˣ)² = σ(x)(1-σ(x))
- d/dx(tanh x) = 1 - tanh²(x)

### Partial Derivatives
- ∂f/∂x: derivative w.r.t. x (hold other variables constant)
- **Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
  - Points in direction of steepest ascent
  - Perpendicular to level curves

### Chain Rule (Multivariate)
If z = f(y) and y = g(x):
- dz/dx = (∂f/∂y)(dy/dx)

For multiple variables:
- ∂z/∂x = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂x)

**Backpropagation is just chain rule!**

### Matrix Calculus
- ∇ₓ(aᵀx) = a
- ∇ₓ(xᵀAx) = (A + Aᵀ)x
- ∇ₓ(||Ax - b||²) = 2Aᵀ(Ax - b)

### Optimization Conditions
- **First-order (necessary)**: ∇f(x*) = 0
- **Second-order (sufficient)**: ∇²f(x*) ≻ 0 (Hessian positive definite)

### Hessian Matrix
H = [∂²f/∂xᵢ∂xⱼ]
- Positive definite → local minimum
- Negative definite → local maximum
- Indefinite → saddle point

---

## PROBABILITY & STATISTICS

### Probability Basics
- **Sample Space** (Ω): All possible outcomes
- **Event** (A): Subset of sample space
- **Probability**: P(A) ∈ [0,1], P(Ω) = 1
- **Addition Rule**: P(A∪B) = P(A) + P(B) - P(A∩B)
- **Multiplication Rule**: P(A∩B) = P(A)P(B|A)
- **Independence**: P(A∩B) = P(A)P(B)

### Conditional Probability & Bayes Theorem
- **Conditional**: P(A|B) = P(A∩B)/P(B)
- **Bayes Theorem**: P(A|B) = P(B|A)P(A)/P(B)
  - P(A|B): posterior
  - P(B|A): likelihood
  - P(A): prior
  - P(B): evidence/normalization

**Python**:
```python
def bayes(prior, likelihood, evidence):
    return (likelihood * prior) / evidence
```

### Random Variables
- **Expectation**: E[X] = Σ x·P(X=x) (discrete) or ∫ x·f(x)dx (continuous)
- **Variance**: Var[X] = E[(X-μ)²] = E[X²] - (E[X])²
- **Standard Deviation**: σ = √Var[X]
- **Covariance**: Cov[X,Y] = E[(X-μₓ)(Y-μᵧ)]
- **Correlation**: ρ = Cov[X,Y]/(σₓσᵧ) ∈ [-1,1]

### Common Distributions

**Bernoulli** (single trial):
- P(X=1) = p, P(X=0) = 1-p
- E[X] = p, Var[X] = p(1-p)

**Binomial** (n trials):
- P(X=k) = C(n,k)pᵏ(1-p)ⁿ⁻ᵏ
- E[X] = np, Var[X] = np(1-p)

**Poisson** (count of rare events):
- P(X=k) = (λᵏe⁻λ)/k!
- E[X] = Var[X] = λ

**Normal/Gaussian**:
- f(x) = (1/√(2πσ²))exp(-(x-μ)²/(2σ²))
- E[X] = μ, Var[X] = σ²
- 68% within 1σ, 95% within 2σ, 99.7% within 3σ

**Exponential** (waiting time):
- f(x) = λe⁻λˣ
- E[X] = 1/λ, Var[X] = 1/λ²

**Python**:
```python
from scipy import stats

# Binomial
stats.binom.pmf(k, n, p)      # P(X=k)
stats.binom.cdf(k, n, p)      # P(X≤k)

# Normal
stats.norm.pdf(x, mu, sigma)  # Probability density
stats.norm.cdf(x, mu, sigma)  # Cumulative probability

# Poisson
stats.poisson.pmf(k, lambda)  # P(X=k)
```

### Maximum Likelihood Estimation (MLE)
**Goal**: Find θ that maximizes P(data|θ)

Steps:
1. Write likelihood: L(θ) = P(x₁,x₂,...,xₙ|θ)
2. Take log: ℓ(θ) = log L(θ) = Σᵢ log P(xᵢ|θ)
3. Differentiate: ∂ℓ/∂θ = 0
4. Solve for θ

**Examples**:
- Normal: μ̂ = (1/n)Σxᵢ, σ̂² = (1/n)Σ(xᵢ-μ̂)²
- Bernoulli: p̂ = k/n
- Poisson: λ̂ = (1/n)Σxᵢ

### Statistical Tests
**Z-score**: z = (x - μ)/σ
**t-statistic**: t = (x̄ - μ)/(s/√n)
**Chi-square**: χ² = Σ(Observed - Expected)²/Expected
**p-value**: P(observing data at least as extreme | H₀ true)

---

## OPTIMIZATION

### Gradient Descent
**Update Rule**: θₜ₊₁ = θₜ - α∇L(θₜ)
- α: learning rate
- ∇L: gradient of loss

**Variants**:
1. **Batch GD**: Use all data
2. **Stochastic GD**: Use one sample
3. **Mini-batch GD**: Use small batch

### Momentum
vₜ₊₁ = βvₜ - α∇L(θₜ)
θₜ₊₁ = θₜ + vₜ₊₁
- β ∈ [0,1): momentum coefficient (typically 0.9)

### Adam (Adaptive Moment Estimation)
```
mₜ = β₁mₜ₋₁ + (1-β₁)∇L(θₜ)         # First moment (mean)
vₜ = β₂vₜ₋₁ + (1-β₂)(∇L(θₜ))²      # Second moment (variance)
m̂ₜ = mₜ/(1-β₁ᵗ)                     # Bias correction
v̂ₜ = vₜ/(1-β₂ᵗ)
θₜ₊₁ = θₜ - α·m̂ₜ/(√v̂ₜ + ε)
```
- β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸ (typical)

### Learning Rate Schedules
- **Constant**: α(t) = α₀
- **Step Decay**: α(t) = α₀·γᵏ (every k epochs)
- **Exponential**: α(t) = α₀e⁻ᵏᵗ
- **1/t**: α(t) = α₀/(1+kt)
- **Cosine Annealing**: α(t) = α_min + 0.5(α_max-α_min)(1+cos(πt/T))

### Convexity
Function f is convex if:
- f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y) for all λ∈[0,1]
- Equivalently: Hessian ∇²f ⪰ 0

**Properties**:
- Local minimum = global minimum
- Gradient descent converges to global minimum

---

## LOSS FUNCTIONS

### Regression
**Mean Squared Error (MSE)**:
- L = (1/n)Σ(yᵢ - ŷᵢ)²
- Gradient: ∂L/∂ŷᵢ = -2(yᵢ - ŷᵢ)/n

**Mean Absolute Error (MAE)**:
- L = (1/n)Σ|yᵢ - ŷᵢ|
- More robust to outliers

**Huber Loss** (smooth MAE):
- L = {½(y-ŷ)²  if |y-ŷ| ≤ δ
      {δ|y-ŷ|-½δ²  otherwise

### Classification
**Binary Cross-Entropy**:
- L = -(1/n)Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
- Gradient: ∂L/∂ŷᵢ = -(yᵢ/ŷᵢ - (1-yᵢ)/(1-ŷᵢ))/n

**Categorical Cross-Entropy** (multi-class):
- L = -(1/n)Σᵢ Σⱼ yᵢⱼlog(ŷᵢⱼ)
- yᵢⱼ: one-hot encoded labels

**Hinge Loss** (SVM):
- L = (1/n)Σ max(0, 1 - yᵢŷᵢ)
- yᵢ ∈ {-1, +1}

---

## INFORMATION THEORY

### Entropy
**Shannon Entropy**: H(X) = -Σᵢ p(xᵢ)log₂(p(xᵢ))
- Measures uncertainty/information content
- Units: bits (log₂) or nats (ln)
- Maximum for uniform distribution: log₂(n)

### Cross-Entropy
H(p,q) = -Σᵢ p(xᵢ)log(q(xᵢ))
- Cost of encoding p using code for q
- Used as loss function in classification

### KL Divergence (Kullback-Leibler)
D_KL(p||q) = Σᵢ p(xᵢ)log(p(xᵢ)/q(xᵢ))
- Measures difference between distributions
- D_KL(p||q) ≥ 0, equals 0 iff p = q
- Not symmetric: D_KL(p||q) ≠ D_KL(q||p)
- Relationship: D_KL(p||q) = H(p,q) - H(p)

### Mutual Information
I(X;Y) = H(X) + H(Y) - H(X,Y)
- Measures dependency between X and Y
- I(X;Y) = 0 ⟺ X and Y independent
- Used in feature selection

**Python**:
```python
from sklearn.metrics import mutual_info_score
mi = mutual_info_score(X, Y)
```

---

## ACTIVATION FUNCTIONS

| Function | Formula | Derivative | Range |
|----------|---------|------------|-------|
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) | (0,1) |
| **Tanh** | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh²(x) | (-1,1) |
| **ReLU** | max(0,x) | {0 if x<0, 1 if x≥0} | [0,∞) |
| **Leaky ReLU** | max(αx,x) | {α if x<0, 1 if x≥0} | (-∞,∞) |
| **Softmax** | eˣⁱ/Σⱼeˣʲ | σᵢ(1-σᵢ) for i=j | (0,1), Σσᵢ=1 |

**When to use**:
- Sigmoid: Binary classification (output layer)
- Tanh: Hidden layers (centered at 0)
- ReLU: Hidden layers (most common, fast)
- Softmax: Multi-class classification (output layer)

---

## GRAPH THEORY

### Basic Concepts
- **Graph**: G = (V,E) - vertices and edges
- **Degree**: Number of edges connected to a node
- **Path**: Sequence of vertices connected by edges
- **Connected**: Path exists between all vertex pairs
- **Cycle**: Path that starts and ends at same vertex

### Representations
**Adjacency Matrix**: A[i,j] = 1 if edge (i,j) exists
**Adjacency List**: dict of neighbors for each node

### Centrality Measures
**Degree Centrality**: Number of connections
- C_D(v) = degree(v)/(n-1)

**Betweenness Centrality**: Fraction of shortest paths through node
- C_B(v) = Σₛ≠ᵥ≠ₜ σₛₜ(v)/σₛₜ

**Eigenvector Centrality**: Importance of neighbors
- Av = λv where A is adjacency matrix

**PageRank**: Iterative importance
- PR(v) = (1-d)/n + d·Σᵤ PR(u)/outdegree(u)

### Graph Neural Networks
**Message Passing**: h_v^(l+1) = σ(W^(l)·AGG({h_u^(l) : u∈N(v)}))
- Aggregate information from neighbors
- Transform with learned weights
- Apply non-linearity

---

## NEURAL NETWORK FUNDAMENTALS

### Forward Pass
```
Layer i: zᵢ = Wᵢaᵢ₋₁ + bᵢ
         aᵢ = σ(zᵢ)
```

### Backpropagation
```
Output layer: δᴸ = ∂L/∂zᴸ
Hidden layer: δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)
Gradients:    ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ
              ∂L/∂bˡ = δˡ
```
(⊙ = element-wise multiplication)

### Weight Initialization
- **Zero**: Bad (symmetry problem)
- **Random**: Normal(0, σ²)
- **Xavier**: σ = √(2/(n_in + n_out))
- **He**: σ = √(2/n_in) - for ReLU

### Regularization
**L1**: L_total = L_data + λ·Σ|wᵢ|
**L2**: L_total = L_data + λ·Σwᵢ²
**Dropout**: Randomly set activations to 0 (p probability)
**Batch Norm**: Normalize layer inputs

---

## COMMON PITFALLS & DEBUG TIPS

### Optimization Issues
❌ **Vanishing Gradients**: Gradients → 0 in deep networks
✓ Fix: ReLU, Batch Norm, Residual connections

❌ **Exploding Gradients**: Gradients → ∞
✓ Fix: Gradient clipping, lower learning rate

❌ **Local Minima**: Stuck in suboptimal solution
✓ Fix: Momentum, Adam, multiple random restarts

❌ **Learning Rate too high**: Divergence
✓ Fix: Reduce α, use learning rate schedule

❌ **Learning Rate too low**: Slow convergence
✓ Fix: Increase α, use adaptive methods (Adam)

### Data Issues
❌ **Unscaled Features**: Different ranges
✓ Fix: StandardScaler, MinMaxScaler

❌ **Imbalanced Classes**: Model ignores minority
✓ Fix: Resampling, class weights, SMOTE

❌ **Overfitting**: High train acc, low test acc
✓ Fix: Regularization, more data, dropout, early stopping

❌ **Underfitting**: Low train and test acc
✓ Fix: More complex model, more features, less regularization

### Implementation Checks
- Always check shapes: print(X.shape)
- Visualize loss curve
- Start with small dataset
- Test gradient computation (finite differences)
- Monitor gradient norms
- Check for NaN/Inf values

---

## USEFUL NUMPY/SCIPY OPERATIONS

```python
import numpy as np
from scipy import stats, linalg

# Linear Algebra
np.dot(A, B)              # Matrix multiplication
np.linalg.inv(A)          # Inverse
np.linalg.eig(A)          # Eigenvalues/vectors
np.linalg.svd(A)          # SVD
np.linalg.norm(v)         # L2 norm
np.linalg.norm(v, ord=1)  # L1 norm

# Statistics
np.mean(X, axis=0)        # Mean along axis
np.std(X, axis=0)         # Standard deviation
np.cov(X.T)               # Covariance matrix
np.corrcoef(X.T)          # Correlation matrix

# Probability
stats.norm.pdf(x, mu, sigma)    # Normal PDF
stats.norm.cdf(x, mu, sigma)    # Normal CDF
stats.binom.pmf(k, n, p)        # Binomial PMF
stats.poisson.pmf(k, lam)       # Poisson PMF

# Optimization
from scipy.optimize import minimize
result = minimize(fun, x0, method='BFGS', jac=gradient)
```

---

## KEY INEQUALITIES

**Cauchy-Schwarz**: |⟨x,y⟩| ≤ ||x|| ||y||
**Triangle**: ||x + y|| ≤ ||x|| + ||y||
**Jensen**: For convex f: f(E[X]) ≤ E[f(X)]
**Markov**: P(X ≥ a) ≤ E[X]/a
**Chebyshev**: P(|X-μ| ≥ kσ) ≤ 1/k²
**Hoeffding**: Concentration inequality for bounded variables

---

## BIOLOGICAL APPLICATIONS CHEAT SHEET

| Problem | Math Concepts | Common Methods |
|---------|--------------|----------------|
| **Gene Expression Analysis** | PCA, SVD, Clustering | Dimensionality reduction, Differential expression |
| **Sequence Alignment** | Dynamic programming, Scoring matrices | Smith-Waterman, BLAST |
| **Protein Structure** | Optimization, Energy minimization | Molecular dynamics, Homology modeling |
| **Population Genetics** | Probability, Hardy-Weinberg | Coalescent theory, Selection models |
| **Phylogenetics** | Graph theory, Maximum likelihood | Neighbor-joining, Maximum parsimony |
| **Drug Discovery** | Regression, Classification, GNNs | QSAR, Virtual screening |
| **Single-cell RNA-seq** | Manifold learning, Clustering | t-SNE, UMAP, Louvain clustering |
| **Protein-Protein Interactions** | Graph theory, Centrality | Network analysis, Community detection |
| **Variant Calling** | Bayesian inference, Hypothesis testing | GATK, FreeBayes |
| **Image Analysis** | CNNs, Computer vision | Cell segmentation, Tissue classification |

---

*Keep this reference handy! Update with your own notes and insights as you learn.*
