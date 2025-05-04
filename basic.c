#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_rhs(int N, double *b) {
    double h = 1.0 / N;
    int num_points = (N-1) * (N-1);
    for (int i = 1; i <= N-1; ++i) {
        for (int j = 1; j <= N-1; ++j) {
            int k = (i-1)*(N-1) + (j-1);
            double x = i * h;
            double y = j * h;
            b[k] = h * h * 2.0 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
        }
    }
}

// Matrix-vector product using stencil (for CG later)
void mat_vec(int N, double h, const double *x, double *Ax) {
    int size = (N-1)*(N-1);
    for (int k = 0; k < size; ++k) {
        int i = k / (N-1) + 1;
        int j = k % (N-1) + 1;
        double sum = 4.0 * x[k];
        if (j > 1) sum -= x[k-1];          // Left
        if (j < N-1) sum -= x[k+1];        // Right
        if (i > 1) sum -= x[k - (N-1)];    // Top
        if (i < N-1) sum -= x[k + (N-1)];  // Bottom
        Ax[k] = sum / (h * h);
    }
}