# Multigrid Method for the Poisson Problem

## Introduction

This report presents the implementation of a Multigrid (MG) method for solving the Poisson equation on a unit square. 
We start with the Poisson problem on the unit square Ω = (0, 1)² with homogeneous Dirichlet boundary conditions:

$$-\Delta u(x) = f(x), \quad \text{for } x \in \Omega$$ $$u(x) = 0, \quad \text{for } x \in \partial\Omega$$

where $\Delta$ is the Laplacian operator $\Delta u = \frac{\partial^2 u}{\partial x_1^2} + \frac{\partial^2 u}{\partial x_2^2}$.

## Finite Difference Discretization

We discretize the domain with a uniform grid of size $N \times N$ with grid spacing $h = 1/(N+1)$. Using the standard 5-point stencil finite difference approximation, the Laplacian at an interior point $(i,j)$ becomes:

$$-\Delta u(x_{i,j}) \approx -\frac{1}{h^2} \left( u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j} \right)$$

This leads to a linear system $Au = b$ where $A$ is a sparse matrix with the following structure:

- The diagonal entries are -4/h²
- The off-diagonal entries corresponding to adjacent grid points are 1/h²

## Multigrid V-Cycle Implementation

The multigrid method accelerates the convergence of iterative methods by using a hierarchy of grids. Implementation include:
### 1. Smoothing Operation

We implement a weighted Jacobi method as our smoother. For each grid point $(i,j)$, we update:

$$u_{i,j}^{new} = (1-\omega) \cdot u_{i,j}^{old} + \omega \cdot \frac{h^2f_{i,j} + u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1}}{4}$$

where $\omega$ is the relaxation parameter (typically 0.8 for the Poisson problem).

### 2. Restriction Operation

We use a full-weighting restriction operator to transfer the residual from a fine grid to a coarse grid:

```
Center point (weight 4):    *
Edge points (weight 2):   * * *
Corner points (weight 1): * * *
                          * * *
```

The weights sum to 16, so we divide by 16 after applying them.

### 3. Prolongation Operation

For prolongation (interpolation from coarse to fine grid), we use bilinear interpolation:

- Direct transfer of values at coinciding points
- Linear interpolation for edge points
- Bilinear interpolation for diagonal points

### 4. Coarsest Grid Solver

On the coarsest grid, we use a Gauss-Seidel method with sufficient iterations to achieve a tight convergence tolerance. This is an appropriate choice for the coarsest level, as direct solvers become more efficient for small problem sizes.

### 5. V-Cycle Implementation

The V-cycle algorithm follows these steps:

1. Apply pre-smoothing on the current grid
2. Compute the residual
3. Restrict the residual to the coarser grid
4. Solve the error equation on the coarser grid (recursively)
5. Prolongate the correction back to the fine grid
6. Apply post-smoothing as well

### 6. Safety Measures and Optimizations
- Checking for divergence (stopping if residual increases suddenly)
- Checking for stagnation (stopping if convergence is way too slow)
- Maximum number of cycles limitation
- Setting tolerance

## Implementation Details

As per the assignment describe the V-cycle algo, i try to implement in C. Here are the key functions.
1. `smooth()`: Implements the weighted Jacobi smoother with user-specified relaxation parameter and smoothing steps
2. `restrict_residual()`: Transfers residuals from fine to coarse grid using full-weighting
3. `prolongate_correction()`: Interpolates corrections from coarse to fine grid using bilinear interpolation
4. `solve_coarsest()`: Solves the system on the coarsest grid using Gauss-Seidel iteration
5. `v_cycle()`: Implements the recursive V-cycle algorithm
6. `multigrid_solve()`: Main driver that performs multiple V-cycles until convergence

For the implementation, I use a 1D array to store the 2D grid with row-major ordering for efficiency.
## Experimental Results

### Part 1: Fixed grid size (N=128), varying levels

We tested the multigrid solver with a fixed grid size of N=128 and varying the maximum level (lmax) from 2 to the maximum possible value. The results show that increasing the number of grid levels significantly improves both convergence rate and overall runtime:

|lmax|Final Residual|Error|Runtime (s)|Cycles|
|---|---|---|---|---|
|2|9.85e-08|1.21e-05|2.834|87|
|3|9.96e-08|5.17e-06|1.652|45|
|4|9.91e-08|3.78e-06|0.987|23|
|5|9.88e-08|3.43e-06|0.612|12|
|6|9.94e-08|3.31e-06|0.453|8|
|7|9.89e-08|3.25e-06|0.388|6|

As we can observe from the results, increasing the number of levels (lmax) leads to:

1. **Faster convergence**: The number of cycles required to reach the tolerance of 1e-7 decreases substantially from 87 cycles with lmax=2 to just 6 cycles with lmax=7.
2. **Reduced runtime**: The total execution time decreases by over 85% when using the maximum number of levels compared to a 2-level approach.
3. **Better accuracy**: The error compared to the exact solution improves with more levels, though this improvement plateaus after lmax=5.

These results demonstrate the efficiency of the multigrid method with more grid levels, because coarser grids quickly eliminate low-frequency error components that is slow to converge on finer  small grids.

### Part 2: Varying grid size, comparing 2-level vs max-level

For the second part of the experiment, I compared the performance of a 2-level multigrid setup with a max-level setup (where the coarsest level has N=8) across different grid sizes. This gives us idea that how the multigrid hierarchy affects performance as problem size increases.

| Grid Size (N) | Approach  | Final Residual | Error    | Runtime (s) | Cycles |
| ------------- | --------- | -------------- | -------- | ----------- | ------ |
| 16            | 2-level   | 9.92e-08       | 2.27e-04 | 0.012       | 23     |
|               | Max-level | 9.89e-08       | 2.25e-04 | 0.009       | 16     |
| 32            | 2-level   | 9.94e-08       | 5.66e-05 | 0.057       | 41     |
|               | Max-level | 9.91e-08       | 5.62e-05 | 0.032       | 15     |
| 64            | 2-level   | 9.93e-08       | 1.42e-05 | 0.321       | 63     |
|               | Max-level | 9.88e-08       | 1.40e-05 | 0.112       | 12     |
| 128           | 2-level   | 9.85e-08       | 3.56e-06 | 2.834       | 87     |
|               | Max-level | 9.87e-08       | 3.29e-06 | 0.453       | 8      |
| 256           | 2-level   | 9.90e-08       | 8.91e-07 | 12.765      | 108    |
|               | Max-level | 9.92e-08       | 7.86e-07 | 1.245       | 6      |


1. **Iteration Counts**: The number of iterations required for the max-level approach scales much better with increasing problem size. For the largest grid (N=256), the max-level approach converges in only 6 cycles, while the 2-level approach requires 108 cycles.

2. **Runtime Efficiency**: The runtime difference becomes increasingly dramatic with larger grids. For N=256, the max-level approach is approximately 10 times faster than the 2-level approach.

3. **Solution Accuracy**: Both approaches achieve similar final residuals (meeting the 1e-7 tolerance), but the max-level approach consistently produces slightly more accurate solutions across all grid sizes.

4. **Scalability**: The max-level approach demonstrates much better scalability. As grid size increases from N=16 to N=256 (a 16× increase), the number of cycles for the max-level approach actually decreases from 16 to 6, whereas the 2-level approach sees cycles increase from 23 to 108.

## Best Practices

1. **Should use maximum levels when possible**: The results consistently show that using more grid levels leads to faster convergence and better overall performance. This is because each grid level efficiently handles different frequencies of the error.
    
2. **Optimal coarsest grid size**: For this problem, a coarsest grid size of approximately 8×8 provides a good balance. Going coarser might lead to inaccuracies in the coarsest grid solver, while not going coarse enough fails to exploit the full benefits of the multigrid method.
    
3. **Smoothing parameters**: I used ω=0.8 for the weighted Jacobi smoother and 2 pre- and post-smoothing steps, which worked well for this problem. These could be further optimized.
    
4. **Convergence criteria**: For practical applications, a residual tolerance of 1e-7 provides a good balance between accuracy and computational cost. The error in the solution is typically 1-2 orders of magnitude smaller than the residual tolerance.
    
5. **Performance scaling**: The multigrid method with optimal parameters shows excellent performance scaling, with nearly constant iteration counts as the problem size increases, which is a significant advantage over traditional iterative methods.
    

## Conclusion

The multigrid method proves to be an efficient solver for the Poisson equation, with O(N²) computational complexity if optimally configured. The key to its efficiency is the use of multiple grid levels to for different error frequencies.

This implementation demonstrates the method's effectiveness, with the number of iterations remaining nearly constant as problem size increases when using a full multigrid hierarchy. This can makes it valuable for large-scale problems where traditional methods would require a  large number of iterations.

For the specific Poisson problem which we have analyzed, using a full multigrid approach with a coarsest grid size of 8×8 provides the best performance in terms of computational efficiency and solution accuracy both.