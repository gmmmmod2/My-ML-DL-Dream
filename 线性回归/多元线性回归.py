import math
import numpy as np
import copy


# Function to calculate the cost using Squared error cost function
def compute_cost(X, y, w, b):
    m = X.shape[0]  # number of training examples
    predictions = X @ w + b  # Matrix multiplication
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def compute_gradient(X, y, w, b):
    m = X.shape[0]  # number of training examples
    predictions = X @ w + b
    error = predictions - y
    dj_dw = (1 / m) * (X.T @ error)  # Gradient for weights
    dj_db = (1 / m) * np.sum(error)  # Gradient for bias
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    b = b_in

    # Arrays to store cost J and parameters w and b at each iteration
    J_history = []
    p_history = []

    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(X, y, w, b)

        # Update Parameters using gradient descent update rule
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))
            p_history.append((w.copy(), b))

        # Print cost at intervals
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw} dj_db: {dj_db}  ",
                  f"w: {w} b:{b}")

    return w, b, J_history, p_history  # return w and J, w history for graphing


# Load our data set
# y = a * x1 +b * x2
X_train = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])  # features
y_train = np.array([2.0, 3.0, 4.0])  # target value

# initialize parameters
w_init = np.zeros(X_train.shape[1])
b_init = 0.0
# some gradient descent settings
iterations = 10000
alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(X_train, y_train, w_init, b_init, alpha, iterations, compute_cost,
                                                    compute_gradient)
print(f"(w, b) found by gradient descent: (w={w_final}, b={b_final})")
