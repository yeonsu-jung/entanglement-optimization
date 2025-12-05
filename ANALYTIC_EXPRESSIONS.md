# Analytic Expressions for C++ Translation

This document provides the key mathematical formulations extracted from the Python implementation, ready for C++ translation.

## Core Algorithms

### 1. Segment-Segment Distance

The fundamental geometric computation for collision detection between rod pairs:

```cpp
double segment_segment_distance(Vec3 p1s, Vec3 p1e, Vec3 p2s, Vec3 p2e)
```

**Mathematical Formulation:**
- Find parameters `t,u ∈ [0,1]` that minimize: `||p1s + t*(p1e-p1s) - p2s - u*(p2e-p2s)||`
- Uses dot products: `D1 = ||d1||²`, `D2 = ||d2||²`, `S1 = d1·d12`, `S2 = d2·d12`, `R = d1·d2`
- Case analysis based on determinant `den = D1*D2 - R²`

**Key Cases:**
1. **Degenerate segments** (one/both are points): Direct projection
2. **Parallel segments** (`den = 0`): Project one onto the other
3. **General case**: Solve linear system with clamping to [0,1]

### 2. Linking Number (Entanglement)

Two equivalent formulations for measuring topological entanglement:

#### Method 1: Gauss Integral (4-term arcsine)
```cpp
double linking_number_gauss(Vec3 p_i, Vec3 p_ii, Vec3 p_j, Vec3 p_jj)
```

**Formula:**
```
LK = -1/(4π) * |Σ arcsin(nᵢ · nᵢ₊₁)|
```
where `nᵢ` are normalized cross products of the quadrilateral edges.

#### Method 2: Arai Formula (2-term arctangent)
```cpp
double linking_number_arai(Vec3 p_i, Vec3 p_ii, Vec3 p_j, Vec3 p_jj)
```

**Formula:**
```
LK = -1/(2π) * |arctan2(a·(b×c), ...) + arctan2(c·(d×a), ...)|
```

More numerically stable for some configurations.

### 3. Rod Representation

**Input Format:** Each rod described by 5 parameters:
- `(x, y, z)`: Center position  
- `(φ, θ)`: Orientation in spherical coordinates

**Conversion to Endpoints:**
```cpp
Vec3 direction = {sin(φ)cos(θ), sin(φ)sin(θ), cos(φ)}
Vec3 endpoint1 = center - 0.5 * length * direction  
Vec3 endpoint2 = center + 0.5 * length * direction
```

### 4. Periodic Boundary Conditions

For simulations in periodic boxes:

```cpp
double segment_segment_distance_pbc(Vec3 p1s, Vec3 p1e, Vec3 p2s, Vec3 p2e, double box)
```

**Algorithm:**
1. Find minimum-image displacement between segment midpoints
2. Shift entire second segment by appropriate lattice vector
3. Compute standard Euclidean distance

### 5. Potential Energy Functions

**Harmonic Repulsion (Collision Avoidance):**
```
U_rep = A * (d - d_collision)²
```

**Combined Potential:**
```
U_total = U_rep + U_entanglement
U_entanglement = linking_number(rod1, rod2)
```

### 6. Gradient Computations

For optimization algorithms (FIRE, L-BFGS, gradient descent), we need analytical gradients with respect to the 5 rod parameters: `(x, y, z, φ, θ)`.

#### Distance Gradient

The gradient of segment-segment distance involves the chain rule:

```
∂d/∂qᵢ = (∂d/∂closest_points) * (∂closest_points/∂endpoints) * (∂endpoints/∂qᵢ)
```

**Key steps:**
1. Find optimal parameters `t, u` (same as distance calculation)
2. Compute closest points: `c₁ = p₁ₛ + t*d₁`, `c₂ = p₂ₛ + u*d₂`
3. Unit separation vector: `û = (c₁ - c₂)/||c₁ - c₂||`
4. Gradients w.r.t. endpoints: `∂d/∂p₁ₛ = û*(1-t)`, `∂d/∂p₁ₑ = û*t`
5. Convert to rod parameters via spherical coordinate derivatives

**Spherical coordinate derivatives:**
```cpp
∂direction/∂φ = (cos(φ)cos(θ), cos(φ)sin(θ), -sin(φ))
∂direction/∂θ = (-sin(φ)sin(θ), sin(φ)cos(θ), 0)
```

#### Linking Number Gradient

The linking number gradient is analytically complex due to transcendental functions. Two approaches:

1. **Full analytical** (very lengthy) - involves derivatives of arcsin/arctan2 
2. **Finite differences** (recommended for initial implementation)

```cpp
// Finite difference approximation
const double eps = 1e-8;
∂LK/∂x ≈ (LK(x + eps) - LK(x)) / eps
```

#### Combined Potential Gradient

```cpp
∇U_total = ∇U_rep + ∇U_entanglement
∇U_rep = 2*A*(d - d_collision) * ∇d
```

### 7. Implementation Strategy

#### Distance Gradients
- **Complexity:** Moderate - involves careful chain rule application
- **Accuracy:** Exact analytical expressions  
- **Performance:** Fast - only basic arithmetic operations

#### Linking Number Gradients  
- **Complexity:** High - transcendental function derivatives
- **Accuracy:** Finite differences adequate for optimization (1e-8 step size)
- **Performance:** Moderate - requires 10 function evaluations per rod pair

#### Validation
Always validate gradients against finite differences:
```cpp
double analytical = grad.dx;
double finite_diff = (f(x + eps) - f(x)) / eps; 
double rel_error = |analytical - finite_diff| / (|finite_diff| + 1e-12);
```
Expect relative errors < 1e-6 for distance gradients, < 1e-4 for linking gradients.

## Implementation Notes

### Numerical Stability
- Use `std::clamp()` for parameter bounds in [0,1]
- Add small tolerance (1e-6) to avoid division by zero
- Clip arcsine arguments to [-1+ε, 1-ε] range

### Performance Considerations
- All operations are O(1) per rod pair
- Main computational cost: sqrt, arcsin, arctan2 functions
- Consider vectorization for large rod ensembles
- Spatial hashing can reduce O(N²) pair evaluations

### Validation
- Distance function should be symmetric and non-negative
- Linking number should be bounded: |LK| ≤ 0.5 for single segment pairs
- Both formulations should give identical results (within numerical precision)

## File Organization

The complete C++ implementation is provided in:
- `cpp_analytic_expressions.hpp`: Header with all function definitions
- Includes utility classes: `Vec3`, `Rod`
- Namespace: `entanglement::`

This provides a direct, efficient translation of the core mathematical algorithms without Python/JAX dependencies.