import numpy as np
import cupy as cp
import time
from dataclasses import dataclass

# Configuration for the algorithm
@dataclass
class PowerIterationConfig:
    dim: int = 4096                    # Matrix size (dim x dim)
    dominance: float = 0.1             # How much larger the top eigenvalue is (controls convergence speed)
    max_steps: int = 400               # Maximum iterations
    check_frequency: int = 10          # Check for convergence every N steps
    progress: bool = True              # Print progress logs
    residual_threshold: float = 1e-10  # Stop if error is below this

def generate_host(cfg=PowerIterationConfig()):
    """Generates a random diagonalizable matrix on the CPU."""
    np.random.seed(42)

    # Create eigenvalues: One large one (1.0), the rest smaller
    weak_lam = np.random.random(cfg.dim - 1) * (1.0 - cfg.dominance)
    lam = np.random.permutation(np.concatenate(([1.0], weak_lam)))

    # Construct matrix A = P * D * P^-1
    P = np.random.random((cfg.dim, cfg.dim))  # Random invertible matrix
    D = np.diag(np.random.permutation(lam))   # Diagonal matrix of eigenvalues
    A = ((P @ D) @ np.linalg.inv(P))          # The final matrix
    return A

# Generate the data on Host
print("Generating Host Data...")
A_host = generate_host()
print(f"Host Matrix Shape: {A_host.shape}")
print(f"Data Type: {A_host.dtype}")


def estimate_host(A, cfg=PowerIterationConfig()):
    """
    Performs power iteration using purely NumPy (CPU).
    """
    # Initialize vector of ones on Host
    x = np.ones(A.shape[0], dtype=np.float64)

    for i in range(0, cfg.max_steps, cfg.check_frequency):
        # Matrix-Vector multiplication
        y = A @ x
        
        # Rayleigh quotient: (x . y) / (x . x)
        lam = (x @ y) / (x @ x)
        
        # Calculate residual (error)
        res = np.linalg.norm(y - lam * x)
        
        # Normalize vector for next step
        x = y / np.linalg.norm(y)

        if cfg.progress:
            print(f"Step {i}: residual = {res:.3e}")

        # Convergence check
        if res < cfg.residual_threshold:
            break

        # Run intermediate steps without checking residual to save compute
        for _ in range(cfg.check_frequency - 1):
            y = A @ x
            x = y / np.linalg.norm(y)

    return (x.T @ (A @ x)) / (x.T @ x)

# Run CPU Baseline
print("\nRunning CPU Estimate...")
start_time = time.time()
lam_est_host = estimate_host(A_host)
end_time = time.time()

print(f"\nEstimated Eigenvalue (CPU): {lam_est_host}")
print(f"Time taken: {end_time - start_time:.4f}s")

def estimate_device_exercise(A, cfg=PowerIterationConfig()):
    """
    TODO: Port the power iteration algorithm to the GPU using CuPy.
    
    Steps to complete:
    1. Transfer the input matrix A to the GPU (if it's a numpy array)
    2. Initialize the vector x on the GPU
    3. Replace np operations with cp operations
    4. Return the result as a Python scalar
    """
    # ---------------------------------------------------------
    # TODO 1: MEMORY TRANSFER (Host -> Device)
    # Check if A is a numpy array. If so, move it to GPU using cp.asarray()
    # Otherwise, assume it's already on the device.
    # ---------------------------------------------------------
    if isinstance(A, np.ndarray):
        A_gpu = cp.asarray(A)  # TODO: Transfer to GPU
    else:
        A_gpu = A
    
    # ---------------------------------------------------------
    # TODO 2: Initialize vector of ones ON THE GPU
    # Hint: Use cp.ones() instead of np.ones()
    # ---------------------------------------------------------
    # x = np.ones(A.shape[0], dtype=np.float64)
    x = cp.ones(A_gpu.shape[0],dtype=cp.float64)  # TODO: Create vector of ones on GPU
    
    for i in range(0, cfg.max_steps, cfg.check_frequency):
        # ---------------------------------------------------------
        # TODO 3: Perform GPU computations
        # Replace the operations below with CuPy equivalents
        # ---------------------------------------------------------
        
        # Matrix-Vector multiplication (this works the same with CuPy!)
        y = A_gpu @ x
        
        # Rayleigh quotient
        lam = (x @ y) / (x @ x)
        
        # TODO: Calculate residual using cp.linalg.norm (not np.linalg.norm)
        # res = np.linalg.norm(y - lam * x)
        res = cp.linalg.norm(y - lam * x)
        
        # TODO: Normalize x using cp.linalg.norm
        x = y / np.linalg.norm(y)
        
        if cfg.progress:
            print(f"Step {i}: residual = {res:.3e}")
        
        if res < cfg.residual_threshold:
            break
        
        for _ in range(cfg.check_frequency - 1):
            y = A_gpu @ x
            x = y / cp.linalg.norm(y)
    
    # ---------------------------------------------------------
    # TODO 4: MEMORY TRANSFER (Device -> Host)
    # Return the eigenvalue as a Python scalar using .item()
    # ---------------------------------------------------------
    result = (x.T @ (A_gpu @ x)) / (x.T @ x)
    return result.item()

# Uncomment to test your implementation:
lam_test = estimate_device_exercise(A_host, PowerIterationConfig(max_steps=50))
print(f"Your result: {lam_test}")

