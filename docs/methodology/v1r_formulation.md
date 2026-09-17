# Mathematical Formulation of STAIR-CNLGCL v1-R

This document presents the rigorous mathematical formulation of **STAIR-CNLGCL v1-R** (Cross-Component Neighborhood-Layer Graph Contrastive Learning with Reweighted Backward Smoothing Convolution). The architecture is structured into four core mathematical modules, formalizing the entire pipeline from raw multimodal features to regularized bipartite representation learning and parameter optimization.

---

## Notational Conventions

| Symbol | Definition | Dimensionality / Domain |
| :--- | :--- | :--- |
| $\mathcal{U}, \mathcal{I}$ | User and Item index sets | $|\mathcal{U}| = N_u$, $|\mathcal{I}| = N_i$ |
| $R \in \{0, 1\}^{N_u \times N_i}$ | Implicit user-item interaction matrix | $R_{ui} = 1$ if $(u, i)$ observed |
| $X^{txt}, X^{vis}$ | Raw extracted item features (Textual, Visual) | $X^{txt} \in \mathbb{R}^{N_i \times d_{txt}}$, $X^{vis} \in \mathbb{R}^{N_i \times d_{vis}}$ |
| $d$ | Latent representation dimension | $d = 64$ |
| $E_u^{(0)}, E_i^{(0)}$ | Initial ego embeddings at layer $0$ | $E_u^{(0)} \in \mathbb{R}^{N_u \times d}$, $E_i^{(0)} \in \mathbb{R}^{N_i \times d}$ |
| $A^{\text{base}} \in \mathbb{R}^{N_i \times N_i}$ | Base item-item multimodal kNN adjacency matrix | Non-negative, zero diagonal ($A_{ii}^{\text{base}} = 0$) |
| $\hat{A} \in \mathbb{R}^{N_i \times N_i}$ | Boosted symmetric normalized adjacency matrix | $\hat{A} = D^{-1/2} W_{\text{sym}} D^{-1/2}$, $\rho(\hat{A}) \le 1.0$ |
| $\tilde{A}_{\text{bi}}$ | Symmetric normalized user-item bipartite adjacency | $\tilde{A}_{\text{bi}} \in \mathbb{R}^{(N_u + N_i) \times (N_u + N_i)}$ |
| $H^{(l)}$ | Concatenated hidden representations at layer $l$ | $H^{(l)} = [E_u^{(l)} \,;\, E_i^{(l)}] \in \mathbb{R}^{(N_u + N_i) \times d}$ |
| $\beta_3 \in \mathbb{R}^d, \beta \in \mathbb{R}^d$ | Spectral filter vectors ($\beta = \mathbf{1} - \beta_3$) | $\beta \in (0, 1)^d$ |
| $\mathcal{L}_{\text{BPR}}, \mathcal{L}_{CL}$ | Recommendation BPR loss and CNLGCL InfoNCE loss | Scalar losses |

---

## Module A — SVD Multimodal Initialization

The objective of Module A is to project heterogeneous, high-dimensional textual and visual embeddings ($d_{txt}, d_{vis} \gg d$) into an isotropic, unit-variance subspace of dimension $d$ while removing dimensional collapse and dominant singular value biases.

### 1. Centering and Truncated SVD
For each modality $m \in \{txt, vis\}$ with raw feature matrix $X^m \in \mathbb{R}^{N_i \times d_m}$, we first compute the mean-centered feature matrix:

$$\bar{X}^m = X^m - \frac{1}{N_i} \mathbf{1}_{N_i} \mathbf{1}_{N_i}^T X^m$$

The thin Singular Value Decomposition (SVD) of $\bar{X}^m$ is defined as:

$$\bar{X}^m = U_m \Sigma_m V_m^T$$

where $U_m \in \mathbb{R}^{N_i \times r_m}$ has orthonormal columns ($U_m^T U_m = I_{r_m}$), $\Sigma_m = \text{diag}(\sigma_{m, 1}, \dots, \sigma_{m, r_m})$ contains singular values in descending order, and $r_m = \min(N_i, d_m)$.

### 2. Isotropic Whitening
We retain the top-$d$ left singular vectors $U_m[:, 1:d]$ and scale them uniformly by $\sqrt{N_i / d}$:

$$\tilde{X}_m = U_m[:, 1:d] \sqrt{\frac{N_i}{d}} \in \mathbb{R}^{N_i \times d}$$

**Property (Exact Covariance Isotropy):**  
The sample covariance matrix of $\tilde{X}_m$ satisfies:

$$\text{Cov}(\tilde{X}_m) = \frac{1}{N_i} \tilde{X}_m^T \tilde{X}_m = \frac{1}{N_i} \left(\sqrt{\frac{N_i}{d}} U_m[:, 1:d]^T\right) \left(\sqrt{\frac{N_i}{d}} U_m[:, 1:d]\right) = \frac{1}{d} I_d$$

This whitening transformation eliminates correlation between latent dimensions, enforces uniform variance $1/d$ across all dimensions, and prevents representation collapse before training begins.

### 3. Multimodal Linear Fusion
The initial item embedding matrix $X_0 \in \mathbb{R}^{N_i \times d}$ is obtained via convex combination with modality confidence weights $k_t, k_v > 0$:

$$X_0 = \frac{k_t \tilde{X}_{txt} + k_v \tilde{X}_{vis}}{k_t + k_v}$$

In practice, $k_t = 5.0$ and $k_v = 1.0$ reflect the higher semantic discriminability of textual descriptions over vision in e-commerce benchmarks.

### 4. Zero-Step User Initialization
Item ego embeddings are initialized directly with $X_0$:

$$E_i^{(0)} = X_0$$

User ego embeddings $E_u^{(0)}$ are initialized by mapping user historical interaction profiles to the multimodal item space through row-normalized collaborative interactions:

$$R_{\text{norm}} = D_u^{-1} R, \quad \text{where } (D_u)_{uu} = \max\left(1, \sum_{j=1}^{N_i} R_{uj}\right)$$

$$E_u^{(0)} = R_{\text{norm}} X_0 = D_u^{-1} R X_0$$

This guarantees that at epoch $0$, each user's representation resides at the centroid of the items they have interacted with in the isotropic multimodal space.

---

## Module B — Consensus-Aware Graph Reweighting (Contribution #2)

Module B constructs a topologically faithful item-item affinity graph $\hat{A}$ for the backward gradient smoother without edge pruning ($0\%$ edge loss), safely bounded to prevent gradient explosion.

### 1. Base Topology and Self-Loop Removal
Let $A^{\text{base}} \in \mathbb{R}^{N_i \times N_i}$ denote the union of mutual $k$-nearest-neighbor item graphs constructed from raw textual and visual features. All self-loops are explicitly eliminated to prevent degenerate diffusion accumulation:

$$A_{ii}^{\text{base}} = 0, \quad \forall i \in \mathcal{I}$$

### 2. Multimodal Semantic Consensus ($q_{ij}^{MM}$)
Let $s_{ij}^{txt} = \cos(x_i^{txt}, x_j^{txt})$ and $s_{ij}^{vis} = \cos(x_i^{vis}, x_j^{vis})$ denote cosine similarities between raw item features. The multimodal consensus metric is defined via thresholded geometric mean:

$$q_{ij}^{MM} = \sqrt{ [s_{ij}^{txt} - \tau_t]_+ \cdot [s_{ij}^{vis} - \tau_v]_+ + \epsilon }$$

where $[z]_+ = \max(0, z)$, $\tau_t, \tau_v \in [0, 1)$ are noise-filtering thresholds, and $\epsilon = 10^{-8}$ is a numerical stabilizer.

*When generalized to $M \ge 3$ modalities (e.g., vision, text, audio on short-video benchmarks), $q_{ij}^{MM}$ computes the average pairwise thresholded consensus:*

$$q_{ij}^{MM} = \frac{2}{M(M-1)} \sum_{1 \le p < q \le M} \sqrt{ [s_{ij}^{(p)} - \tau_p]_+ \cdot [s_{ij}^{(q)} - \tau_q]_+ + \epsilon }$$

### 3. Collaborative Behavioral Consensus ($q_{ij}^{CF}$)
Behavioral consensus measures structural co-interaction strength using the Ochiai similarity coefficient on the user interaction bipartite graph:

$$q_{ij}^{CF} = \frac{|U_i \cap U_j|}{\sqrt{d_i \cdot d_j} + \epsilon} = \frac{(R^T R)_{ij}}{\sqrt{d_i \cdot d_j} + \epsilon}$$

where $U_i = \{u \in \mathcal{U} \mid R_{ui} = 1\}$, $d_i = |U_i| = \sum_{u=1}^{N_u} R_{ui}$ is the interaction degree of item $i$, and $(R^T R)_{ij}$ counts common users who interacted with both items $i$ and $j$.

### 4. Multiplicative Safe Boost and Clamping
The composite edge weight $W_{ij}$ multiplicatively augments the base topological connection:

$$W_{ij} = A_{ij}^{\text{base}} \cdot \left(1 + \alpha \cdot q_{ij}^{MM} + \beta \cdot q_{ij}^{CF}\right)$$

where $\alpha \ge 0, \beta \ge 0$ control semantic and collaborative reinforcement strengths. To prevent gradient shock on dense hubs, weights are strictly clamped within a safe interval $[w_{\min}, w_{\max}]$:

$$W_{ij}' = \text{clip}\left(W_{ij}, w_{\min}, w_{\max}\right) = \max\left(w_{\min}, \min(W_{ij}, w_{\max})\right)$$

with $w_{\min} = 1.0$ and $w_{\max} = 3.6$.

### 5. Symmetrization and Symmetric Normalized Adjacency
The directed boosted graph is symmetrized via maximum operator to preserve bidirectional edge consensus:

$$W_{\text{sym}} = \max\left(W', (W')^T\right), \quad \text{i.e., } (W_{\text{sym}})_{ij} = \max(W_{ij}', W_{ji}')$$

The degree matrix $D \in \mathbb{R}^{N_i \times N_i}$ is defined as $D_{ii} = \sum_{j=1}^{N_i} (W_{\text{sym}})_{ij}$. The resulting graph operator is the **Symmetric Normalized Adjacency Matrix**:

$$\boxed{ \hat{A} = D^{-1/2} W_{\text{sym}} D^{-1/2} }$$

$$\hat{A}_{ij} = \frac{(W_{\text{sym}})_{ij}}{\sqrt{D_{ii} D_{jj}}}$$

#### Mathematical Distinction: Adjacency vs. Laplacian
> **Rigorous Analytical Note:**  
> $\hat{A} = D^{-1/2} W_{\text{sym}} D^{-1/2}$ is the **Symmetric Normalized Adjacency Matrix**, NOT the normalized Laplacian.  
> The Normalized Graph Laplacian is defined as $L = I - D^{-1/2} W_{\text{sym}} D^{-1/2} = I - \hat{A}$.  
> - While $L$ is Symmetric Positive Semi-Definite (SPSD) with spectrum $\sigma(L) \subseteq [0, 2]$,  
> - $\hat{A}$ has all its eigenvalues bounded in $[-1, 1]$ by the Gershgorin circle theorem, yielding a spectral radius:
> 
> $$\rho(\hat{A}) = \max_{\lambda \in \sigma(\hat{A})} |\lambda| \le 1.0$$
> 
> This bounded spectral radius guarantees that the matrix power series $\sum_{l=0}^L \beta_3^l \hat{A}^l$ used in backward gradient smoothing converges stably and cannot cause gradient explosion.

---

## Module C — Stepwise Propagation (FSC Backbone)

Module C implements Forward Stepwise Convolution (FSC) on the bipartite user-item graph, retaining intermediate representation snapshots for contrastive regularization.

### 1. Unified Embedding State
The unified representation tensor at layer $0$ stacks user and item embeddings:

$$H^{(0)} = \begin{bmatrix} E_u^{(0)} \\ E_i^{(0)} \end{bmatrix} \in \mathbb{R}^{(N_u + N_i) \times d}$$

### 2. Normalized Bipartite Adjacency
Let the symmetric normalized bipartite adjacency matrix $\tilde{A}_{\text{bi}} \in \mathbb{R}^{(N_u + N_i) \times (N_u + N_i)}$ be structured as:

$$\tilde{A}_{\text{bi}} = \begin{bmatrix} 0 & D_u^{-1/2} R D_i^{-1/2} \\ D_i^{-1/2} R^T D_u^{-1/2} & 0 \end{bmatrix}$$

where $D_u = \text{diag}(\sum_j R_{uj})$ and $D_i = \text{diag}(\sum_u R_{ui})$.

### 3. Stepwise Layer Propagation
The frequency-aware decay vector $\beta \in \mathbb{R}^d$ is derived from the model spectral parameter $\beta_3 \in \mathbb{R}^d$:

$$\beta_{3, k} = 0.1 + 0.9 \left(\frac{k}{d}\right)^\gamma, \quad k \in \{0, 1, \dots, d-1\}, \; \gamma = 0.2$$

$$\beta = \mathbf{1} - \beta_3 \in (0, 1)^d$$

For each layer $l \in \{0, 1, \dots, L-1\}$ (with $L = 3$):

$$H^{(l+1)} = \left(\tilde{A}_{\text{bi}} H^{(l)}\right) \odot \beta$$

where $\odot$ denotes element-wise Hadamard broadcasting along feature dimensions.

### 4. Neumann-Like Accumulated Representation
The final representations for top-$K$ ranking are aggregated across all $L$ layers via normalized polynomial summation:

$$H^* = \left( \sum_{l=0}^{L} H^{(l)} \right) \odot \frac{\mathbf{1} - \beta}{\mathbf{1} - \beta^{L+1}}$$

The final user and item embeddings are retrieved as:

$$E_u^*, E_i^* = \text{split}\left(H^*, [N_u, N_i]\right)$$

**Key Architecture Storage:**  
The model explicitly caches the sequence of unaggregated hidden representations $\mathcal{H} = \{H^{(0)}, H^{(1)}, \dots, H^{(L)}\}$. Specifically, $H^{(0)}$ and $H^{(1)}$ serve as direct inputs to Module D.

---

## Module D — Cross-Layer Neighborhood-Aware Contrastive Regularization (CNLGCL, Contribution #1)

Module D constitutes **Contribution #1**: a cross-layer contrastive regularizer that enforces representation uniformity and alignment between ego representations ($H^{(0)}$) and first-order aggregated neighborhood representations ($H^{(1)}$) **without any parameterized projection head**, ensuring $100\%$ gradient backpropagation directly into the embedding tables $E_u, E_i$.

```
Layer 0 (Ego Embeddings):         U_0 = H^(0)[users]         I_0 = H^(0)[pos_items]
                                        │                           │
                              Cross-Layer Alignment       Cross-Layer Alignment
                                        │                           │
                                        ▼                           ▼
Layer 1 (Neighborhood Aggregation): I_1 = H^(1)[pos_items]   U_1 = H^(1)[users]
                                   [Direction 1: U -> I]      [Direction 2: I -> U]
```

### 1. Representation Extraction (No Projection Head)
For each mini-batch $\mathcal{B}$ with user indices $u \in \mathcal{B}_u$ and positive item indices $i \in \mathcal{B}_i$ ($|\mathcal{B}| = B$):

$$U_0, I_0 = \text{split}(H^{(0)}, [N_u, N_i]), \quad U_1, I_1 = \text{split}(H^{(1)}, [N_u, N_i])$$

$$u_0 = U_0[u] \in \mathbb{R}^{B \times d}, \quad i_1 = I_1[i] \in \mathbb{R}^{B \times d}$$

$$i_0 = I_0[i] \in \mathbb{R}^{B \times d}, \quad u_1 = U_1[u] \in \mathbb{R}^{B \times d}$$

### 2. Implementation Component 1: True Sign-Preserving Spectral Perturbation
To prevent representation collapse while preserving the topological quadrant of the feature space, a sign-preserving perturbation is injected into any hidden vector $h$:

$$\tilde{h} = h + \epsilon \cdot \left(\beta \odot \text{sgn}(h) \odot \frac{|\eta|}{\| |\eta| \|_2}\right), \quad \eta \sim \mathcal{N}(0, I_d)$$

where $|\eta| \ge 0$ is the element-wise absolute value of Gaussian noise, $\text{sgn}(\cdot)$ is the signum function, and $\epsilon = 0.08$.

**Mathematical Invariant:**  
Because $|\eta| \ge 0$ and $\beta > 0$, the perturbation term has the exact same sign as $h$ along every coordinate:

$$\text{sgn}\left(\epsilon \cdot \beta_k \cdot \text{sgn}(h_k) \cdot \frac{|\eta_k|}{\| |\eta| \|_2}\right) = \text{sgn}(h_k) \quad (\forall h_k \ne 0)$$

Thus, the perturbation strictly expands or contracts coordinates radially without quadrant flips or topological distortions.

### 3. Implementation Component 2: Fused Operations on Tensor $[4, B, d]$
To optimize GPU execution, the four mini-batch vectors are stacked into a single contiguous tensor:

$$T_{\text{fused}} = \text{stack}\left([u_0, i_1, i_0, u_1], \text{dim}=0\right) \in \mathbb{R}^{4 \times B \times d}$$

Perturbation and $L_2$ sphere projection are applied simultaneously along the last dimension:

$$\tilde{T}_{\text{fused}} = \text{Normalize}_{L_2}\left(T_{\text{fused}} + \epsilon \cdot \beta \odot \text{sgn}(T_{\text{fused}}) \odot \frac{|\eta|}{\| |\eta| \|_2}\right)$$

$$[\tilde{u}_0, \tilde{i}_1, \tilde{i}_0, \tilde{u}_1] = \text{unbind}\left(\tilde{T}_{\text{fused}}, \text{dim}=0\right)$$

This reduces 8 separate CUDA kernel invocations to 2, accelerating training by $11.5\% - 21.4\%$.

### 4. Implementation Component 3: Adaptive Multimodal Margin (AMM)
Let $c_i = \cos(\tilde{x}_i^{txt}, \tilde{x}_i^{vis})$ denote the raw cosine consistency between whitened textual and visual features of item $i$. We normalize consistency over the dataset using the 5th and 95th percentiles ($Q_{0.05}$ and $Q_{0.95}$):

$$\hat{c}_i = \text{clip}\left(\frac{c_i - Q_{0.05}(c)}{Q_{0.95}(c) - Q_{0.05}(c) + 10^{-8}}, 0.0, 1.0\right)$$

The adaptive margin $m_i$ for positive pair $(u_b, i_b)$ is inversely proportional to multimodal agreement:

$$m_i = m_{\max} \cdot (1 - \hat{c}_i)$$

where $m_{\max} = 0.02$ ($10\%$ of temperature $\tau = 0.20$).  
- **High Consistency ($\hat{c}_i \to 1$):** $m_i \to 0$, tight margin avoiding over-separation of reliable embeddings.  
- **Low Consistency / High Conflict ($\hat{c}_i \to 0$):** $m_i \to m_{\max}$, widening the angular separation required between user ego and positive neighborhood.

### 5. Implementation Component 4: In-Batch False Negative Filtering (FNF) Mask
Items sharing high modal similarity with the positive target within the mini-batch should not be penalized as negative samples. The binary mask $M \in \{0, 1\}^{B \times B}$ is formulated as:

$$M_{bk} = \begin{cases}
0, & \text{if } b = k \text{ (diagonal self-pair)} \\
0, & \text{if } \cos(x_{i_b}^{\text{modal}}, x_{i_k}^{\text{modal}}) > \tau_{\text{thresh}} \text{ (semantic false negative)} \\
1, & \text{otherwise (valid negative)}
\end{cases}$$

where $\tau_{\text{thresh}} = 0.85$ (or $0.50$ on fine-grained datasets like Baby).

### 6. Implementation Component 5: Bidirectional InfoNCE with Linear Warmup

#### Direction 1: User-to-Item ($U_0 \to I_1$, with AMM and FNF)
$$\mathcal{L}_{u \to i} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\left( (\tilde{u}_{0, b} \cdot \tilde{i}_{1, b} - m_{i_b}) / \tau \right)}{\exp\left( (\tilde{u}_{0, b} \cdot \tilde{i}_{1, b} - m_{i_b}) / \tau \right) + \sum_{k=1}^{B} M_{bk} \exp\left( (\tilde{u}_{0, b} \cdot \tilde{i}_{1, k}) / \tau \right)}$$

#### Direction 2: Item-to-User ($I_0 \to U_1$, with FNF, no AMM)
Because users lack intrinsic multimodal features, AMM is omitted in the reverse direction:

$$\mathcal{L}_{i \to u} = - \frac{1}{B} \sum_{b=1}^{B} \log \frac{\exp\left( (\tilde{i}_{0, b} \cdot \tilde{u}_{1, b}) / \tau \right)}{\exp\left( (\tilde{i}_{0, b} \cdot \tilde{u}_{1, b}) / \tau \right) + \sum_{k=1}^{B} M_{kb} \exp\left( (\tilde{i}_{0, b} \cdot \tilde{u}_{1, k}) / \tau \right)}$$

#### Composite CNLGCL Loss
$$\mathcal{L}_{CL} = \alpha_d \mathcal{L}_{u \to i} + (1 - \alpha_d) \mathcal{L}_{i \to u}$$

with directional balance parameter $\alpha_d = 0.50$.

#### Linear Warmup Scheduler
To allow BPR loss to establish global manifold structure before contrastive dispersion takes full effect, $\lambda_{\text{cl}}(t)$ follows a linear warmup schedule over $T_{\text{warmup}}$ epochs:

$$\lambda_{\text{cl}}(t) = \begin{cases}
\lambda_{\text{target}} \cdot \dfrac{t}{T_{\text{warmup}}}, & \text{if } t \le T_{\text{warmup}} \\[8pt]
\lambda_{\text{target}}, & \text{if } t > T_{\text{warmup}}
\end{cases}$$

where $\lambda_{\text{target}} = 0.008$ ($0.005$ on large-scale datasets) and $T_{\text{warmup}} = 50$ (or $100$).

---

## Overall Optimization and Backward Smoother

The total objective optimized during the forward pass is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BPR}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}} + \lambda_{\text{cl}}(t) \mathcal{L}_{CL}$$

where:

$$\mathcal{L}_{\text{BPR}} = - \frac{1}{|\mathcal{B}|} \sum_{(u, i, j) \in \mathcal{B}} \log \sigma\left( \hat{y}_{ui} - \hat{y}_{uj} \right), \quad \hat{y}_{ui} = E_u^* \cdot E_i^*$$

$$\mathcal{L}_{\text{reg}} = \frac{1}{2 |\mathcal{B}|} \sum_{(u, i, j) \in \mathcal{B}} \left( \|E_u^{(0)}\|_2^2 + \|E_i^{(0)}\|_2^2 + \|E_j^{(0)}\|_2^2 \right)$$

### Backward Optimization with BSC Smoother
During backpropagation, AdamWSEvo computes base parameter updates $\Delta_t = \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon_{\text{opt}})$ from gradient $\nabla_\theta \mathcal{L}_{\text{total}}$. Item embedding updates are filtered through the consensus-boosted adjacency operator $\hat{A}$ via Neumann series:

$$\tilde{\Delta}_t^{\text{item}} = \text{Smoother}(\Delta_t^{\text{item}}, \hat{A}) = \frac{1 - \beta_3}{1 - \beta_3^{L+1}} \sum_{l=0}^{L} \beta_3^l \hat{A}^l \Delta_t^{\text{item}}$$

$$\theta_{t+1} = \theta_t (1 - \eta \cdot \lambda_{\text{wd}}) - \eta \tilde{\Delta}_t$$

Because $\hat{A}$ satisfies $\rho(\hat{A}) \le 1.0$ and $\beta_3 < 1.0$, the smoother acts as a low-pass graph spectral filter on parameter updates, ensuring orthogonal synergy between forward contrastive alignment and backward topological diffusion.
