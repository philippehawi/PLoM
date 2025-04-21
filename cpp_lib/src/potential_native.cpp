#include <vector>
#include <cmath>
#include <iostream>

// Define export macros for platform compatibility
#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT __attribute__((visibility("default")))
#endif

extern "C" EXPORT void get_L(const double* H, const double* u, double* pot, int nu, int N) {
    // Scaling factors
    double s = std::pow(4.0 / (N * (2.0 + nu)), 1.0 / (nu + 4));
    double shat = s / std::sqrt(s * s + (N - 1.0) / N);
    double norm_factor = -0.5 / (shat * shat);

    // Scale H and store it in a scaled_H vector
    std::vector<double> scaled_H(nu * N);
    for (int i = 0; i < nu; ++i) {
        for (int j = 0; j < N; ++j) {
            scaled_H[i * N + j] = H[i * N + j] * (shat / s);
        }
    }

    // Main computation for all columns of u
    for (int k = 0; k < N; ++k) {
        // Initialize accumulators for dq and q
        std::vector<double> dq(nu, 0.0);
        double q = 0.0;

        for (int j = 0; j < N; ++j) {
            // Compute the squared norm of the difference
            double norm_squared = 0.0;
            for (int i = 0; i < nu; ++i) {
                double diff = scaled_H[i * N + j] - u[i * N + k];
                norm_squared += diff * diff;
            }

            // Compute the weight
            double weight = std::exp(norm_factor * norm_squared);
            q += weight;

            // Accumulate dq
            for (int i = 0; i < nu; ++i) {
                dq[i] += weight * (scaled_H[i * N + j] - u[i * N + k]);
            }
        }

        // Normalize dq and store the result directly in pot
        for (int i = 0; i < nu; ++i) {
            pot[i * N + k] = dq[i] / (q * shat * shat);
        }
    }
}