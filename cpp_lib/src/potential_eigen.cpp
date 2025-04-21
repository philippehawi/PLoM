#include <Eigen/Dense>
#include <cmath>
#include <iostream>  // For debug output

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

    // Precompute constant for norm calculation
    double norm_factor = -0.5 / (shat * shat);

    // Map input arrays H and u to Eigen matrices
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> H_mat(H, nu, N);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> u_mat(u, nu, N);
    Eigen::MatrixXd pot_temp(nu, N);  // Temporary matrix for computation
	
	// Scale H
	Eigen::MatrixXd scaled_H = H_mat * (shat / s);

	// Main computation for all columns of u

	for (int k = 0; k < N; ++k) {
		// Compute pairwise differences: diff = scaled_H - u_col
		Eigen::MatrixXd diff = scaled_H.colwise() - u_mat.col(k);

		// Compute weights for all columns of H
		Eigen::RowVectorXd weights = (-0.5 / (shat * shat) * diff.array().square().colwise().sum()).exp();

		// Compute dq as weighted sum of distances
		Eigen::VectorXd dq = diff * weights.transpose();

		// Compute q as sum of weights
		double q = weights.sum();

		// Normalize and store result
		pot_temp.col(k) = dq / (q * shat * shat);
	}

    // Map pot (output) and store the transposed result
    Eigen::Map<Eigen::MatrixXd> pot_mat(pot, N, nu);  // Map as N x nu
    pot_mat = pot_temp.transpose();  // Store the transpose
}
