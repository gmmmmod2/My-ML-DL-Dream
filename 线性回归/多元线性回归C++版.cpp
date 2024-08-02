#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

// Function to calculate the cost using Squared error cost function
double compute_cost(const MatrixXd& X, const VectorXd& y, const VectorXd& w, double b) {
    int m = X.rows();
    VectorXd predictions = X * w + VectorXd::Constant(m, b);
    VectorXd errors = predictions - y;
    double cost = (1.0 / (2 * m)) * errors.squaredNorm();
    return cost;
}

// Function to calculate gradients
void compute_gradient(const MatrixXd& X, const VectorXd& y, const VectorXd& w, double b, VectorXd& dj_dw, double& dj_db) {
    int m = X.rows();
    VectorXd predictions = X * w + VectorXd::Constant(m, b);
    VectorXd errors = predictions - y;
    dj_dw = (1.0 / m) * X.transpose() * errors;
    dj_db = (1.0 / m) * errors.sum();
}

// Gradient descent function
void gradient_descent(const MatrixXd& X, const VectorXd& y, VectorXd& w, double& b, double alpha, int num_iters) {
    int m = X.rows();
    int n = X.cols();
    vector<double> J_history;
    VectorXd dj_dw(n);
    double dj_db;

    for (int i = 0; i < num_iters; ++i) {
        // Compute the gradient
        compute_gradient(X, y, w, b, dj_dw, dj_db);

        // Update the parameters
        w = w - alpha * dj_dw;
        b = b - alpha * dj_db;

        // Compute and record the cost
        double cost = compute_cost(X, y, w, b);
        J_history.push_back(cost);

        // Print cost at intervals
        if (i % (num_iters / 10) == 0) {
            cout << "Iteration " << i << ": Cost " << cost << endl;
        }
    }
}

int main() {
    // Load our data set
    MatrixXd X_train(3, 2);
    X_train << 1.0, 1.0,
        2.0, 1.0,
        3.0, 1.0;
    VectorXd y_train(3);
    y_train << 2.0, 3.0, 4.0;

    // Initialize parameters
    VectorXd w = VectorXd::Zero(2);
    double b = 0.0;

    // Gradient descent settings
    int iterations = 10000;
    double alpha = 1.0e-2;

    // Run gradient descent
    gradient_descent(X_train, y_train, w, b, alpha, iterations);

    cout << "(w, b) found by gradient descent: (w = " << w.transpose() << ", b = " << b << ")" << endl;

    return 0;
}
