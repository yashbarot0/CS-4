#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

// Function prototypes
void initialize_grid(double *grid, int n);
void initialize_rhs(double *rhs, int n);
double compute_residual(double *u, double *f, int n);
void smooth(double *u, double *f, int n, double omega, int nu);
void restrict_residual(double *r_fine, double *r_coarse, int n_fine);
void prolongate_correction(double *e_coarse, double *e_fine, int n_coarse);
void add_correction(double *u, double *correction, int n);
void solve_coarsest(double *u, double *f, int n);
void print_grid(double *grid, int n);
void print_convergence(int cycle, double residual);
double exact_solution(double x, double y);
double source_function(double x, double y);

// Multigrid V-cycle implementation
void v_cycle(double *u, double *f, int n, double omega, int nu, int lmax, int level) {
    // Base case: solve exactly on the coarsest grid
    if (level == lmax) {
        solve_coarsest(u, f, n);
        return;
    }
    
    int size = n * n;
    double *residual = (double *)malloc(size * sizeof(double));
    
    // Pre-smoothing: nu iterations of weighted Jacobi
    smooth(u, f, n, omega, nu);
    
    // Compute residual: r = f - Au
    for (int i = 0; i < size; i++) {
        residual[i] = 0.0;
    }
    
    double h = 1.0 / (n + 1);
    double h2 = h * h;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            double Au = 0.0;
            
            // Interior points
            if (i > 0) Au += u[(i-1) * n + j];
            if (i < n-1) Au += u[(i+1) * n + j];
            if (j > 0) Au += u[i * n + j-1];
            if (j < n-1) Au += u[i * n + j+1];
            
            // Center point with appropriate scaling
            Au -= 4.0 * u[idx];
            
            // Scale by 1/h^2
            Au /= -h2;
            
            // Residual = f - Au
            residual[idx] = f[idx] - Au;
        }
    }
    
    // Restrict residual to coarser grid
    int n_coarse = (n - 1) / 2;
    double *f_coarse = (double *)malloc(n_coarse * n_coarse * sizeof(double));
    double *u_coarse = (double *)malloc(n_coarse * n_coarse * sizeof(double));
    
    // Initialize coarse grid correction to zero
    for (int i = 0; i < n_coarse * n_coarse; i++) {
        u_coarse[i] = 0.0;
    }
    
    // Restriction operation (fine to coarse)
    restrict_residual(residual, f_coarse, n);
    
    // Recursive call to solve on coarser grid
    v_cycle(u_coarse, f_coarse, n_coarse, omega, nu, lmax, level + 1);
    
    // Prolongate correction back to fine grid
    double *correction = (double *)malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        correction[i] = 0.0;
    }
    
    // Prolongation operation (coarse to fine)
    prolongate_correction(u_coarse, correction, n_coarse);
    
    // Add correction to solution
    add_correction(u, correction, n);
    
    // Post-smoothing: nu iterations of weighted Jacobi
    smooth(u, f, n, omega, nu);
    
    // Clean up
    free(residual);
    free(f_coarse);
    free(u_coarse);
    free(correction);
}

// Full multigrid solver
double multigrid_solve(double *u, double *f, int n, double omega, int nu, int lmax, 
                       int max_cycles, double tol) {
    double residual, initial_residual;
    int cycle = 0;
    
    // Calculate initial residual
    initial_residual = compute_residual(u, f, n);
    residual = initial_residual;
    
    printf("Initial residual: %e\n", residual);
    
    // Main loop
    while (cycle < max_cycles && residual > tol) {
        // Perform one V-cycle
        v_cycle(u, f, n, omega, nu, lmax, 1);
        
        // Compute new residual
        residual = compute_residual(u, f, n);
        
        // Print convergence information
        print_convergence(++cycle, residual);
        
        // Check for divergence
        if (residual > 1e6 * initial_residual) {
            printf("Divergence detected! Stopping.\n");
            break;
        }
        
        // Check for very slow convergence
        if (cycle > 10 && residual > 0.95 * initial_residual) {
            printf("Convergence too slow! Stopping.\n");
            break;
        }
    }
    
    return residual;
}

// Compute L2 norm of the residual
double compute_residual(double *u, double *f, int n) {
    double residual_norm = 0.0;
    double h = 1.0 / (n + 1);
    double h2 = h * h;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            double Au = 0.0;
            
            // Interior points
            if (i > 0) Au += u[(i-1) * n + j];
            if (i < n-1) Au += u[(i+1) * n + j];
            if (j > 0) Au += u[i * n + j-1];
            if (j < n-1) Au += u[i * n + j+1];
            
            // Center point
            Au -= 4.0 * u[idx];
            
            // Scale by 1/h^2
            Au /= -h2;
            
            // Residual = f - Au
            double res = f[idx] - Au;
            residual_norm += res * res;
        }
    }
    
    return sqrt(residual_norm);
}

// Weighted Jacobi smoothing
void smooth(double *u, double *f, int n, double omega, int nu) {
    double h = 1.0 / (n + 1);
    double h2 = h * h;
    double *u_new = (double *)malloc(n * n * sizeof(double));
    
    for (int iter = 0; iter < nu; iter++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                double sum = 0.0;
                
                // Sum of neighboring points
                if (i > 0) sum += u[(i-1) * n + j];
                if (i < n-1) sum += u[(i+1) * n + j];
                if (j > 0) sum += u[i * n + j-1];
                if (j < n-1) sum += u[i * n + j+1];
                
                // Weighted Jacobi update
                u_new[idx] = (1.0 - omega) * u[idx] + omega * (h2 * f[idx] + sum) / 4.0;
            }
        }
        
        // Copy updated values back to u
        for (int i = 0; i < n * n; i++) {
            u[i] = u_new[i];
        }
    }
    
    free(u_new);
}

// Restriction operation (fine to coarse grid)
void restrict_residual(double *r_fine, double *r_coarse, int n_fine) {
    int n_coarse = (n_fine - 1) / 2;
    
    for (int i = 0; i < n_coarse; i++) {
        for (int j = 0; j < n_coarse; j++) {
            int i_fine = 2 * i + 1;
            int j_fine = 2 * j + 1;
            
            // Full-weighting restriction
            double sum = 0.0;
            
            // Center point (weight 4)
            sum += 4.0 * r_fine[i_fine * n_fine + j_fine];
            
            // Edge points (weight 2)
            sum += 2.0 * r_fine[(i_fine-1) * n_fine + j_fine];
            sum += 2.0 * r_fine[(i_fine+1) * n_fine + j_fine];
            sum += 2.0 * r_fine[i_fine * n_fine + (j_fine-1)];
            sum += 2.0 * r_fine[i_fine * n_fine + (j_fine+1)];
            
            // Corner points (weight 1)
            sum += r_fine[(i_fine-1) * n_fine + (j_fine-1)];
            sum += r_fine[(i_fine-1) * n_fine + (j_fine+1)];
            sum += r_fine[(i_fine+1) * n_fine + (j_fine-1)];
            sum += r_fine[(i_fine+1) * n_fine + (j_fine+1)];
            
            // Scale by 1/16
            r_coarse[i * n_coarse + j] = sum / 16.0;
        }
    }
}

// Prolongation operation (coarse to fine grid)
void prolongate_correction(double *e_coarse, double *e_fine, int n_coarse) {
    int n_fine = 2 * n_coarse + 1;
    
    // Set fine grid points to zero first
    for (int i = 0; i < n_fine * n_fine; i++) {
        e_fine[i] = 0.0;
    }
    
    // Bilinear interpolation
    for (int i_c = 0; i_c < n_coarse; i_c++) {
        for (int j_c = 0; j_c < n_coarse; j_c++) {
            int i_f = 2 * i_c + 1;
            int j_f = 2 * j_c + 1;
            
            // Coarse grid point directly to fine grid
            e_fine[i_f * n_fine + j_f] = e_coarse[i_c * n_coarse + j_c];
            
            // Horizontally adjacent points
            if (j_c > 0) {
                e_fine[i_f * n_fine + (j_f-1)] = 0.5 * (e_coarse[i_c * n_coarse + j_c] + 
                                                       e_coarse[i_c * n_coarse + (j_c-1)]);
            }
            
            if (j_c < n_coarse-1) {
                e_fine[i_f * n_fine + (j_f+1)] = 0.5 * (e_coarse[i_c * n_coarse + j_c] + 
                                                       e_coarse[i_c * n_coarse + (j_c+1)]);
            }
            
            // Vertically adjacent points
            if (i_c > 0) {
                e_fine[(i_f-1) * n_fine + j_f] = 0.5 * (e_coarse[i_c * n_coarse + j_c] + 
                                                       e_coarse[(i_c-1) * n_coarse + j_c]);
            }
            
            if (i_c < n_coarse-1) {
                e_fine[(i_f+1) * n_fine + j_f] = 0.5 * (e_coarse[i_c * n_coarse + j_c] + 
                                                       e_coarse[(i_c+1) * n_coarse + j_c]);
            }
            
            // Diagonally adjacent points
            if (i_c > 0 && j_c > 0) {
                e_fine[(i_f-1) * n_fine + (j_f-1)] = 0.25 * (e_coarse[i_c * n_coarse + j_c] + 
                                                            e_coarse[(i_c-1) * n_coarse + j_c] +
                                                            e_coarse[i_c * n_coarse + (j_c-1)] +
                                                            e_coarse[(i_c-1) * n_coarse + (j_c-1)]);
            }
            
            if (i_c > 0 && j_c < n_coarse-1) {
                e_fine[(i_f-1) * n_fine + (j_f+1)] = 0.25 * (e_coarse[i_c * n_coarse + j_c] + 
                                                            e_coarse[(i_c-1) * n_coarse + j_c] +
                                                            e_coarse[i_c * n_coarse + (j_c+1)] +
                                                            e_coarse[(i_c-1) * n_coarse + (j_c+1)]);
            }
            
            if (i_c < n_coarse-1 && j_c > 0) {
                e_fine[(i_f+1) * n_fine + (j_f-1)] = 0.25 * (e_coarse[i_c * n_coarse + j_c] + 
                                                            e_coarse[(i_c+1) * n_coarse + j_c] +
                                                            e_coarse[i_c * n_coarse + (j_c-1)] +
                                                            e_coarse[(i_c+1) * n_coarse + (j_c-1)]);
            }
            
            if (i_c < n_coarse-1 && j_c < n_coarse-1) {
                e_fine[(i_f+1) * n_fine + (j_f+1)] = 0.25 * (e_coarse[i_c * n_coarse + j_c] + 
                                                            e_coarse[(i_c+1) * n_coarse + j_c] +
                                                            e_coarse[i_c * n_coarse + (j_c+1)] +
                                                            e_coarse[(i_c+1) * n_coarse + (j_c+1)]);
            }
        }
    }
}

// Add correction to solution
void add_correction(double *u, double *correction, int n) {
    for (int i = 0; i < n * n; i++) {
        u[i] += correction[i];
    }
}

// Direct solver for the coarsest grid
void solve_coarsest(double *u, double *f, int n) {
    // For very small grids, use simple Gauss-Seidel iteration
    double h = 1.0 / (n + 1);
    double h2 = h * h;
    double residual, tol = 1e-10;
    int max_iter = 1000, iter = 0;
    
    do {
        residual = 0.0;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                double old_val = u[idx];
                double sum = 0.0;
                
                // Sum of neighboring points
                if (i > 0) sum += u[(i-1) * n + j];
                if (i < n-1) sum += u[(i+1) * n + j];
                if (j > 0) sum += u[i * n + j-1];
                if (j < n-1) sum += u[i * n + j+1];
                
                // Gauss-Seidel update
                u[idx] = (h2 * f[idx] + sum) / 4.0;
                
                // Residual computation
                double res = u[idx] - old_val;
                residual += res * res;
            }
        }
        
        residual = sqrt(residual);
        iter++;
        
    } while (iter < max_iter && residual > tol);
}

// Initialize grid to zeros
void initialize_grid(double *grid, int n) {
    for (int i = 0; i < n * n; i++) {
        grid[i] = 0.0;
    }
}

// Initialize right-hand side with f(x,y) = 2π²sin(πx)sin(πy)
void initialize_rhs(double *rhs, int n) {
    double h = 1.0 / (n + 1);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            rhs[i * n + j] = source_function(x, y);
        }
    }
}

// Source function f(x,y) = 2π²sin(πx)sin(πy)
double source_function(double x, double y) {
    return 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
}

// Exact solution u(x,y) = sin(πx)sin(πy)
double exact_solution(double x, double y) {
    return sin(M_PI * x) * sin(M_PI * y);
}

// Print grid values
void print_grid(double *grid, int n) {
    if (n > 16) {
        printf("Grid too large to print (n = %d)\n", n);
        return;
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.6f ", grid[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Print convergence information
void print_convergence(int cycle, double residual) {
    printf("Cycle %3d: Residual = %e\n", cycle, residual);
}

// Calculate error compared to exact solution
double calculate_error(double *u, int n) {
    double error = 0.0;
    double h = 1.0 / (n + 1);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double x = (i + 1) * h;
            double y = (j + 1) * h;
            double exact = exact_solution(x, y);
            double diff = u[i * n + j] - exact;
            error += diff * diff;
        }
    }
    
    return sqrt(error) / (n * n);
}

// Main function
int main() {
    // Test parameters
    int N[] = {16, 32, 64, 128, 256};
    int num_N = sizeof(N) / sizeof(N[0]);
    double omega = 0.8;  // Relaxation parameter for weighted Jacobi
    int nu = 2;         // Number of pre/post-smoothing steps
    double tol = 1e-7;  // Stopping tolerance
    int max_cycles = 100; // Maximum number of V-cycles
    
    printf("Multigrid Solver for the Poisson Equation\n");
    printf("----------------------------------------\n\n");
    
    // Part 1: Fixed grid size, varying lmax
    printf("Part 1: Fixed grid size (N=128), varying lmax\n");
    printf("--------------------------------------------\n");
    
    int fixed_N = 128;
    int max_lmax = 1;
    int temp_N = fixed_N;
    
    // Determine maximum possible lmax
    while (temp_N > 2) {
        temp_N = (temp_N - 1) / 2;
        max_lmax++;
    }
    
    printf("Maximum possible lmax for N=%d is %d\n\n", fixed_N, max_lmax);
    
    // Loop over different lmax values
    for (int lmax = 2; lmax <= max_lmax; lmax++) {
        double *u = (double *)malloc(fixed_N * fixed_N * sizeof(double));
        double *f = (double *)malloc(fixed_N * fixed_N * sizeof(double));
        
        initialize_grid(u, fixed_N);
        initialize_rhs(f, fixed_N);
        
        printf("Running with lmax = %d\n", lmax);
        
        clock_t start = clock();
        double final_res = multigrid_solve(u, f, fixed_N, omega, nu, lmax, max_cycles, tol);
        clock_t end = clock();
        
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;
        double error = calculate_error(u, fixed_N);
        
        printf("Final residual: %e\n", final_res);
        printf("Error: %e\n", error);
        printf("Runtime: %.3f seconds\n\n", runtime);
        
        free(u);
        free(f);
    }
    
    // Part 2: Varying grid size, comparing 2-level vs max-level
    printf("Part 2: Varying grid size, comparing 2-level vs max-level\n");
    printf("-----------------------------------------------------\n");
    
    for (int i = 0; i < num_N; i++) {
        int n = N[i];
        double *u1 = (double *)malloc(n * n * sizeof(double));
        double *f1 = (double *)malloc(n * n * sizeof(double));
        double *u2 = (double *)malloc(n * n * sizeof(double));
        double *f2 = (double *)malloc(n * n * sizeof(double));
        
        // Determine max lmax for this grid size
        int curr_lmax = 1;
        int temp_n = n;
        while (temp_n > 8) {  // Coarsest level has N=8
            temp_n = (temp_n - 1) / 2;
            curr_lmax++;
        }
        
        // Initialize grids
        initialize_grid(u1, n);
        initialize_rhs(f1, n);
        initialize_grid(u2, n);
        initialize_rhs(f2, n);
        
        printf("Grid size N = %d\n", n);
        
        // 2-level multigrid
        printf("2-level multigrid:\n");
        clock_t start1 = clock();
        double res1 = multigrid_solve(u1, f1, n, omega, nu, 2, max_cycles, tol);
        clock_t end1 = clock();
        double runtime1 = (double)(end1 - start1) / CLOCKS_PER_SEC;
        double error1 = calculate_error(u1, n);
        
        // Max-level multigrid
        printf("\nMax-level multigrid (lmax = %d):\n", curr_lmax);
        clock_t start2 = clock();
        double res2 = multigrid_solve(u2, f2, n, omega, nu, curr_lmax, max_cycles, tol);
        clock_t end2 = clock();
        double runtime2 = (double)(end2 - start2) / CLOCKS_PER_SEC;
        double error2 = calculate_error(u2, n);
        
        // Print comparison
        printf("\nComparison summary for N = %d:\n", n);
        printf("                   2-level     Max-level\n");
        printf("Final residual:    %e  %e\n", res1, res2);
        printf("Error:             %e  %e\n", error1, error2);
        printf("Runtime (seconds): %.3f        %.3f\n\n", runtime1, runtime2);
        
        free(u1);
        free(f1);
        free(u2);
        free(f2);
    }
    
    return 0;
}