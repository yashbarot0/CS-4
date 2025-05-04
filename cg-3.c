#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Efficient matrix-vector product for A = (N - |i-j|)/N
void mat_vec(int N, const double *x, double *Ax) {
    double *S = (double *)malloc((N+1) * sizeof(double)); // Prefix sum of x
    double *T = (double *)malloc((N+1) * sizeof(double)); // Prefix sum of j*x_j

    S[0] = 0.0;
    T[0] = 0.0;
    for (int j = 1; j <= N; j++) {
        S[j] = S[j-1] + x[j-1];
        T[j] = T[j-1] + j * x[j-1];
    }

    for (int i = 1; i <= N; i++) {
        double term1 = i * (2 * S[i] - S[N]);
        double term2 = 2 * T[i] - T[N];
        Ax[i-1] = S[N] - (term1 - term2) / N;
    }

    free(S);
    free(T);
}

// CG solver with residual tracking
void cg_solve(int N, const double *b, double *x, double reltol, double abstol,
              int *iterations, double **residuals, int *num_residuals) {
    int max_iter = 10000;
    double *r = malloc(N * sizeof(double));
    double *p = malloc(N * sizeof(double));
    double *Ap = malloc(N * sizeof(double));
    *residuals = malloc(max_iter * sizeof(double));

    memset(x, 0, N * sizeof(double));
    memcpy(r, b, N * sizeof(double));

    double rsold = 0.0;
    for (int i = 0; i < N; i++) rsold += r[i] * r[i];
    double r_norm0 = sqrt(rsold);
    double tol = fmax(reltol * r_norm0, abstol);

    if (r_norm0 < tol) {
        *iterations = 0;
        *num_residuals = 0;
        free(r); free(p); free(Ap);
        return;
    }

    memcpy(p, r, N * sizeof(double));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        mat_vec(N, p, Ap);

        double pAp = 0.0;
        for (int i = 0; i < N; i++) pAp += p[i] * Ap[i];
        double alpha = rsold / pAp;

        for (int i = 0; i < N; i++) x[i] += alpha * p[i];
        for (int i = 0; i < N; i++) r[i] -= alpha * Ap[i];

        double rsnew = 0.0;
        for (int i = 0; i < N; i++) rsnew += r[i] * r[i];
        double r_norm = sqrt(rsnew);
        (*residuals)[iter] = r_norm;

        if (r_norm <= tol) {
            iter++;
            break;
        }

        double beta = rsnew / rsold;
        for (int i = 0; i < N; i++) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    *iterations = iter;
    *num_residuals = iter;

    free(r); free(p); free(Ap);
}

int main() {
    int N_values[] = {100, 1000, 10000, 100000};
    int num_N = sizeof(N_values) / sizeof(N_values[0]);
    double reltol = sqrt(__DBL_EPSILON__);
    double abstol = 0.0;

    for (int i = 0; i < num_N; i++) {
        int N = N_values[i];
        double *b = malloc(N * sizeof(double));
        double *x = malloc(N * sizeof(double));
        for (int j = 0; j < N; j++) b[j] = 1.0;

        double *residuals;
        int iterations, num_residuals;
        cg_solve(N, b, x, reltol, abstol, &iterations, &residuals, &num_residuals);

        // Output residuals to file for plotting
        char filename[50];
        sprintf(filename, "residuals_N%d.txt", N);
        FILE *fp = fopen(filename, "w");
        for (int k = 0; k < num_residuals; k++) {
            fprintf(fp, "%d %.15e\n", k+1, residuals[k]);
        }
        fclose(fp);

        printf("N=%d: iterations=%d\n", N, iterations);
        free(b);
        free(x);
        free(residuals);
    }

    return 0;
}