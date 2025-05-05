/**
 * Multigrid Method Implementation for Poisson Problem
 * Case Study 4 - MAP55672 (2024-25)
 * 
 * This program implements a V-cycle multigrid solver for the Poisson equation:
 * -Δu(x) = f(x) on the unit square domain with zero boundary conditions.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 #include <string.h>
 
 // Function prototypes
 double** create_grid(int n);
 void free_grid(double** grid, int n);
 double** create_matrix(int n);
 void setup_poisson_matrix(double** A, int n);
 void setup_rhs(double** b, int n, double h);
 double compute_residual_norm(double** A, double** x, double** b, int n);
 void weighted_jacobi_smoother(double** A, double** x, double** b, int n, double omega, int nu);
 void restrict_residual(double** r_fine, double** r_coarse, int n_fine);
 void prolongate_and_correct(double** x_fine, double** x_coarse, int n_coarse);
 void copy_grid(double** dst, double** src, int n);
 int v_cycle(double** A, double** x, double** b, double omega, int nu, int l, int lmax, double** A_levels, double** r_levels, double** x_levels);
 void print_grid(double** grid, int n);
 void gauss_seidel_solve(double** A, double** x, double** b, int n, double tol);
 double f_func(double x1, double x2);
 double exact_solution(double x1, double x2);
 void compute_error(double** u, int n, double h);
 
 int main(int argc, char* argv[]) {
     // Parse command line arguments
     if (argc < 4) {
         printf("Usage: %s N lmax max_cycles\n", argv[0]);
         printf("  N: Number of interior grid points in each dimension\n");
         printf("  lmax: Maximum multigrid level (2 or more)\n");
         printf("  max_cycles: Maximum number of V-cycles to perform\n");
         return 1;
     }
     
     int N = atoi(argv[1]);
     int lmax = atoi(argv[2]);
     int max_cycles = atoi(argv[3]);
     
     // Validate input
     if (N <= 0 || (N & (N-1)) != 0) {
         printf("Error: N must be a positive power of 2\n");
         return 1;
     }
     
     if (lmax < 2) {
         printf("Error: lmax must be at least 2\n");
         return 1;
     }
     
     int min_coarse_grid = 4;  // Minimum size of coarsest grid (2 interior points in each dim)
     int N_coarsest = N >> (lmax - 1);
     
     if (N_coarsest < min_coarse_grid) {
         printf("Error: With N=%d and lmax=%d, coarsest grid would be too small (N=%d)\n", N, lmax, N_coarsest);
         printf("Maximum valid lmax for N=%d is %d\n", N, (int)(log2(N) - log2(min_coarse_grid) + 1));
         return 1;
     }
     
     // Mesh spacing
     double h = 1.0 / (N + 1);
     
     // MG parameters
     double omega = 2.0/3.0;  // Relaxation parameter for weighted Jacobi
     int nu = 3;              // Number of pre/post-smoothing steps
     double tol = 1e-7;       // Convergence tolerance for residual norm
     
     // Allocate memory for all grid levels
     double*** A_levels = (double***)malloc(lmax * sizeof(double**));
     double*** x_levels = (double***)malloc(lmax * sizeof(double**));
     double*** r_levels = (double***)malloc(lmax * sizeof(double**));
     
     int n_level = N;
     for (int l = 0; l < lmax; l++) {
         A_levels[l] = create_matrix(n_level);
         x_levels[l] = create_grid(n_level);
         r_levels[l] = create_grid(n_level);
         
         // Setup Poisson matrix for this level
         setup_poisson_matrix(A_levels[l], n_level);
         
         // Move to next coarser level
         n_level /= 2;
     }
     
     // Setup right-hand side for finest level
     setup_rhs(r_levels[0], N, h);
     
     // Initial guess (all zeros, already set by create_grid)
     
     printf("Starting Multigrid solver with:\n");
     printf("  N = %d\n", N);
     printf("  lmax = %d\n", lmax);
     printf("  max_cycles = %d\n", max_cycles);
     printf("  omega = %f\n", omega);
     printf("  nu = %d\n", nu);
     printf("  tolerance = %e\n", tol);
     
     // Start timing
     clock_t start = clock();
     
     // Main iteration loop
     int cycle;
     double initial_residual = compute_residual_norm(A_levels[0], x_levels[0], r_levels[0], N);
     double prev_residual = initial_residual;
     printf("Initial residual: %e\n", initial_residual);
     
     for (cycle = 0; cycle < max_cycles; cycle++) {
         // Perform one V-cycle
         int coarse_solves = v_cycle(A_levels[0], x_levels[0], r_levels[0], omega, nu, 0, lmax-1, A_levels, r_levels, x_levels);
         
         // Compute new residual
         double residual = compute_residual_norm(A_levels[0], x_levels[0], r_levels[0], N);
         
         printf("Cycle %d: residual = %e, reduction = %e, coarse solves = %d\n", 
                cycle+1, residual, residual/prev_residual, coarse_solves);
         
         // Check for convergence
         if (residual < tol) {
             printf("Converged to tolerance after %d cycles!\n", cycle+1);
             break;
         }
         
         // Check for divergence or stagnation
         if (residual > 10.0 * prev_residual) {
             printf("Warning: Solution is diverging, stopping.\n");
             break;
         }
         
         if (residual > 0.9 * prev_residual && cycle > 5) {
             printf("Warning: Very slow convergence, stopping.\n");
             break;
         }
         
         prev_residual = residual;
     }
     
     // End timing
     clock_t end = clock();
     double cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;
     
     printf("Total runtime: %.6f seconds\n", cpu_time);
     
     // Compute and print error against analytical solution
     compute_error(x_levels[0], N, h);
     
     // Free memory
     for (int l = 0; l < lmax; l++) {
         free_grid(A_levels[l], N >> l);
         free_grid(x_levels[l], N >> l);
         free_grid(r_levels[l], N >> l);
     }
     free(A_levels);
     free(x_levels);
     free(r_levels);
     
     return 0;
 }
 
 /**
  * Creates a 2D grid of size n×n, initialized to zero
  */
 double** create_grid(int n) {
     double** grid = (double**)malloc(n * sizeof(double*));
     for (int i = 0; i < n; i++) {
         grid[i] = (double*)calloc(n, sizeof(double));
     }
     return grid;
 }
 
 /**
  * Frees memory allocated for a 2D grid
  */
 void free_grid(double** grid, int n) {
     for (int i = 0; i < n; i++) {
         free(grid[i]);
     }
     free(grid);
 }
 
 /**
  * Creates a matrix for storing the coefficient matrix A
  */
 double** create_matrix(int n) {
     return create_grid(n * n);
 }
 
 /**
  * Sets up the coefficient matrix A for the Poisson equation using
  * finite difference approximation
  */
 void setup_poisson_matrix(double** A, int n) {
     int size = n * n;
     
     // Clear matrix
     for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
             A[i][j] = 0.0;
         }
     }
     
     // Fill matrix with 5-point stencil values
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             int idx = i * n + j;
             
             // Diagonal element
             A[idx][idx] = 4.0;
             
             // Off-diagonal elements (neighbors)
             if (i > 0) A[idx][idx-n] = -1.0;  // up
             if (i < n-1) A[idx][idx+n] = -1.0;  // down
             if (j > 0) A[idx][idx-1] = -1.0;  // left
             if (j < n-1) A[idx][idx+1] = -1.0;  // right
         }
     }
 }
 
 /**
  * Source function for the Poisson equation: f(x) = 2π² sin(πx₁) sin(πx₂)
  */
 double f_func(double x1, double x2) {
     return 2.0 * M_PI * M_PI * sin(M_PI * x1) * sin(M_PI * x2);
 }
 
 /**
  * Exact solution u(x) = sin(πx₁) sin(πx₂)
  */
 double exact_solution(double x1, double x2) {
     return sin(M_PI * x1) * sin(M_PI * x2);
 }
 
 /**
  * Sets up the right-hand side vector b based on the source function
  */
 void setup_rhs(double** b, int n, double h) {
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             // Convert grid index to physical coordinates
             double x1 = (i + 1) * h;
             double x2 = (j + 1) * h;
             
             // Set RHS value from source function and scale by h²
             b[i][j] = f_func(x1, x2) * h * h;
         }
     }
 }
 
 /**
  * Computes the 2-norm of the residual r = b - Ax
  */
 double compute_residual_norm(double** A, double** x, double** b, int n) {
     double norm = 0.0;
     
     // Compute residual directly using the 5-point stencil
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             double Ax_val = 4.0 * x[i][j];
             
             if (i > 0) Ax_val -= x[i-1][j];
             if (i < n-1) Ax_val -= x[i+1][j];
             if (j > 0) Ax_val -= x[i][j-1];
             if (j < n-1) Ax_val -= x[i][j+1];
             
             double residual = b[i][j] - Ax_val;
             norm += residual * residual;
         }
     }
     
     return sqrt(norm);
 }
 
 /**
  * Performs weighted Jacobi smoothing iterations
  */
 void weighted_jacobi_smoother(double** A, double** x, double** b, int n, double omega, int nu) {
     double** x_new = create_grid(n);
     
     for (int iter = 0; iter < nu; iter++) {
         // Perform one Jacobi iteration
         for (int i = 0; i < n; i++) {
             for (int j = 0; j < n; j++) {
                 double sum = 0.0;
                 
                 // Add contributions from neighbors
                 if (i > 0) sum += x[i-1][j];
                 if (i < n-1) sum += x[i+1][j];
                 if (j > 0) sum += x[i][j-1];
                 if (j < n-1) sum += x[i][j+1];
                 
                 // Update using weighted average
                 x_new[i][j] = (1.0 - omega) * x[i][j] + omega * (b[i][j] + sum) / 4.0;
             }
         }
         
         // Copy new values back to x
         copy_grid(x, x_new, n);
     }
     
     free_grid(x_new, n);
 }
 
 /**
  * Restricts the residual from fine to coarse grid (Full weighting)
  */
 void restrict_residual(double** r_fine, double** r_coarse, int n_fine) {
     int n_coarse = n_fine / 2;
     
     for (int i = 0; i < n_coarse; i++) {
         for (int j = 0; j < n_coarse; j++) {
             int i_fine = 2 * i;
             int j_fine = 2 * j;
             
             // Full weighting restriction
             r_coarse[i][j] = 0.25 * r_fine[i_fine][j_fine] +
                              0.125 * (r_fine[i_fine+1][j_fine] + r_fine[i_fine-1][j_fine] + 
                                      r_fine[i_fine][j_fine+1] + r_fine[i_fine][j_fine-1]) +
                              0.0625 * (r_fine[i_fine+1][j_fine+1] + r_fine[i_fine+1][j_fine-1] + 
                                       r_fine[i_fine-1][j_fine+1] + r_fine[i_fine-1][j_fine-1]);
         }
     }
 }
 
 /**
  * Prolongates the correction from coarse to fine grid and adds it to the fine grid solution
  */
 void prolongate_and_correct(double** x_fine, double** x_coarse, int n_coarse) {
     int n_fine = 2 * n_coarse;
     
     // Bilinear interpolation
     for (int i = 0; i < n_fine; i++) {
         for (int j = 0; j < n_fine; j++) {
             int i_coarse = i / 2;
             int j_coarse = j / 2;
             
             // Determine weights based on position
             double w_i = (i % 2 == 0) ? 1.0 : 0.5;
             double w_j = (j % 2 == 0) ? 1.0 : 0.5;
             
             // Apply bilinear interpolation
             if (i % 2 == 0 && j % 2 == 0) {
                 // Direct injection at coarse grid points
                 x_fine[i][j] += x_coarse[i_coarse][j_coarse];
             } 
             else if (i % 2 == 1 && j % 2 == 0) {
                 // Interpolate in i direction
                 if (i_coarse + 1 < n_coarse) {
                     x_fine[i][j] += 0.5 * (x_coarse[i_coarse][j_coarse] + x_coarse[i_coarse+1][j_coarse]);
                 } else {
                     x_fine[i][j] += x_coarse[i_coarse][j_coarse];
                 }
             }
             else if (i % 2 == 0 && j % 2 == 1) {
                 // Interpolate in j direction
                 if (j_coarse + 1 < n_coarse) {
                     x_fine[i][j] += 0.5 * (x_coarse[i_coarse][j_coarse] + x_coarse[i_coarse][j_coarse+1]);
                 } else {
                     x_fine[i][j] += x_coarse[i_coarse][j_coarse];
                 }
             }
             else {
                 // Interpolate in both directions
                 if (i_coarse + 1 < n_coarse && j_coarse + 1 < n_coarse) {
                     x_fine[i][j] += 0.25 * (x_coarse[i_coarse][j_coarse] + 
                                            x_coarse[i_coarse+1][j_coarse] +
                                            x_coarse[i_coarse][j_coarse+1] +
                                            x_coarse[i_coarse+1][j_coarse+1]);
                 } else if (i_coarse + 1 < n_coarse) {
                     x_fine[i][j] += 0.5 * (x_coarse[i_coarse][j_coarse] + 
                                           x_coarse[i_coarse+1][j_coarse]);
                 } else if (j_coarse + 1 < n_coarse) {
                     x_fine[i][j] += 0.5 * (x_coarse[i_coarse][j_coarse] + 
                                           x_coarse[i_coarse][j_coarse+1]);
                 } else {
                     x_fine[i][j] += x_coarse[i_coarse][j_coarse];
                 }
             }
         }
     }
 }
 
 /**
  * Copies values from src grid to dst grid
  */
 void copy_grid(double** dst, double** src, int n) {
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             dst[i][j] = src[i][j];
         }
     }
 }
 
 /**
  * Solves the coarsest level system using Gauss-Seidel iteration
  */
 void gauss_seidel_solve(double** A, double** x, double** b, int n, double tol) {
     int max_iter = 1000;
     double** r = create_grid(n);
     
     for (int iter = 0; iter < max_iter; iter++) {
         // Perform one Gauss-Seidel iteration
         for (int i = 0; i < n; i++) {
             for (int j = 0; j < n; j++) {
                 double sum = 0.0;
                 
                 // Add contributions from neighbors (using updated values where available)
                 if (i > 0) sum += x[i-1][j];
                 if (i < n-1) sum += x[i+1][j];
                 if (j > 0) sum += x[i][j-1];
                 if (j < n-1) sum += x[i][j+1];
                 
                 // Update solution
                 x[i][j] = (b[i][j] + sum) / 4.0;
             }
         }
         
         // Check residual every few iterations
         if (iter % 5 == 0) {
             double res_norm = compute_residual_norm(A, x, b, n);
             if (res_norm < tol) {
                 break;
             }
         }
     }
     
     free_grid(r, n);
 }
 
 /**
  * Recursive implementation of the V-cycle multigrid algorithm
  * Returns the number of coarse level solves performed
  */
 int v_cycle(double** A, double** x, double** b, double omega, int nu, int l, int lmax, 
            double*** A_levels, double*** r_levels, double*** x_levels) {
     int n = 1 << (log2(r_levels[0][0] != 0 ? sizeof(r_levels[0])/sizeof(r_levels[0][0]) : 1) - l);
     int coarse_solves = 0;
     
     // 1. Pre-smoothing
     weighted_jacobi_smoother(A, x, b, n, omega, nu);
     
     // 2. Compute residual r = b - A*x
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             double Ax_val = 4.0 * x[i][j];
             
             if (i > 0) Ax_val -= x[i-1][j];
             if (i < n-1) Ax_val -= x[i+1][j];
             if (j > 0) Ax_val -= x[i][j-1];
             if (j < n-1) Ax_val -= x[i][j+1];
             
             r_levels[l][i][j] = b[i][j] - Ax_val;
         }
     }
     
     // 3. Restrict residual to coarser grid
     if (l < lmax) {
         int n_coarse = n / 2;
         double** r_coarse = r_levels[l+1];
         
         // Reset coarse grid correction
         for (int i = 0; i < n_coarse; i++) {
             for (int j = 0; j < n_coarse; j++) {
                 x_levels[l+1][i][j] = 0.0;
             }
         }
         
         // Restrict residual
         restrict_residual(r_levels[l], r_coarse, n);
         
         // 4. Recursively solve coarse grid problem or solve directly if at coarsest level
         if (l+1 == lmax) {
             // Direct solve at coarsest level
             gauss_seidel_solve(A_levels[l+1], x_levels[l+1], r_coarse, n_coarse, 1e-10);
             coarse_solves = 1;
         } else {
             // Recurse to next level
             coarse_solves = v_cycle(A_levels[l+1], x_levels[l+1], r_coarse, omega, nu, l+1, lmax, 
                                   A_levels, r_levels, x_levels);
         }
         
         // 5. Prolongate and correct
         prolongate_and_correct(x, x_levels[l+1], n_coarse);
     }
     
     // 6. Post-smoothing
     weighted_jacobi_smoother(A, x, b, n, omega, nu);
     
     return coarse_solves;
 }
 
 /**
  * Prints grid values for debugging
  */
 void print_grid(double** grid, int n) {
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             printf("%8.5f ", grid[i][j]);
         }
         printf("\n");
     }
 }
 
 /**
  * Computes the error between numerical and exact solution
  */
 void compute_error(double** u, int n, double h) {
     double max_error = 0.0;
     double l2_error = 0.0;
     
     for (int i = 0; i < n; i++) {
         for (int j = 0; j < n; j++) {
             // Convert grid index to physical coordinates
             double x1 = (i + 1) * h;
             double x2 = (j + 1) * h;
             
             // Compute exact solution
             double u_exact = exact_solution(x1, x2);
             
             // Compute error
             double error = fabs(u[i][j] - u_exact);
             l2_error += error * error;
             
             if (error > max_error) {
                 max_error = error;
             }
         }
     }
     
     l2_error = sqrt(l2_error / (n * n));
     
     printf("Error analysis:\n");
     printf("  Max error: %e\n", max_error);
     printf("  L2 error:  %e\n", l2_error);
 }