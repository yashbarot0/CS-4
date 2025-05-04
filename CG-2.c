#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// Generate the right-hand side vector b
void generate_rhs(int N, double *b) {
    double h = 1.0 / N;
    int size = (N-1) * (N-1);
    for (int i = 1; i <= N-1; ++i) {
        for (int j = 1; j <= N-1; ++j) {
            int k = (i-1)*(N-1) + (j-1);
            double x = i * h;
            double y = j * h;
            b[k] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

// Matrix-vector product A*x using the 5-point stencil
void mat_vec(int N, double h, const double *x, double *Ax) {
    int size = (N-1) * (N-1);
    for (int k = 0; k < size; ++k) {
        int i = k / (N-1) + 1;
        int j = k % (N-1) + 1;
        double sum = 4.0 * x[k];
        if (j > 1) sum -= x[k-1];          // Left neighbor
        if (j < N-1) sum -= x[k+1];        // Right neighbor
        if (i > 1) sum -= x[k - (N-1)];    // Top neighbor
        if (i < N-1) sum -= x[k + (N-1)];  // Bottom neighbor
        Ax[k] = sum / (h * h);
    }
}

// CG solver for Ax = b
void cg_solve(int N, double *b, double *x, double tol, int *iterations, double *time_used) {
    int max_iter = 10000;
    int size = (N-1)*(N-1);
    double *r = malloc(size * sizeof(double));
    double *p = malloc(size * sizeof(double));
    double *Ap = malloc(size * sizeof(double));

    memset(x, 0, size * sizeof(double));
    memcpy(r, b, size * sizeof(double));

    double rsold = 0.0;
    for (int i = 0; i < size; ++i) rsold += r[i] * r[i];
    double residual_norm = sqrt(rsold);

    if (residual_norm < tol) {
        *iterations = 0;
        *time_used = 0.0;
        free(r); free(p); free(Ap);
        return;
    }

    memcpy(p, r, size * sizeof(double));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        mat_vec(N, 1.0/N, p, Ap);

        double pAp = 0.0;
        for (int i = 0; i < size; ++i) pAp += p[i] * Ap[i];
        double alpha = rsold / pAp;

        for (int i = 0; i < size; ++i) x[i] += alpha * p[i];
        for (int i = 0; i < size; ++i) r[i] -= alpha * Ap[i];

        double rsnew = 0.0;
        for (int i = 0; i < size; ++i) rsnew += r[i] * r[i];
        residual_norm = sqrt(rsnew);

        if (residual_norm < tol) {
            iter++;
            break;
        }

        double beta = rsnew / rsold;
        for (int i = 0; i < size; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    gettimeofday(&end, NULL);
    *time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
    *iterations = iter;

    free(r); free(p); free(Ap);
}

int main() {
    int N_values[] = {8, 16, 32, 64, 128, 256};
    int num_N = sizeof(N_values) / sizeof(N_values[0]);

    printf("N\tIterations\tTime (s)\n");
    for (int i = 0; i < num_N; ++i) {
        int N = N_values[i];
        int size = (N-1)*(N-1);
        double *b = malloc(size * sizeof(double));
        double *x = malloc(size * sizeof(double));

        generate_rhs(N, b);
        int iterations;
        double time_used;
        cg_solve(N, b, x, 1e-8, &iterations, &time_used);

        printf("%d\t%d\t\t%.6f\n", N, iterations, time_used);

        free(b);
        free(x);
    }
    return 0;
}