"""
NeuroCore - Perceptron Logic Module
Implements activation functions and neural network calculations
"""

import numpy as np


def activation_threshold(z, threshold=0):
    """
    Threshold Function (Hard-limit)
    Returns 1 if z >= threshold, else 0
    """
    return np.where(z >= threshold, 1, 0)


def activation_sigmoid(z, threshold=0, gain=1.0):
    """
    Sigmoid Function
    f(z) = 1 / (1 + e^(-a*(z - threshold)))
    where a is the gain parameter
    """
    z_shifted = z - threshold
    return 1 / (1 + np.exp(-gain * z_shifted))


def calculate_net_input(features, weights):
    """
    Calculate the net input: sum(x_i * w_i)
    """
    return np.dot(features, weights)


def perceptron_predict(features, weights, activation_type='threshold', threshold=0, gain=1.0):
    """
    Main perceptron prediction function

    Parameters:
    - features: array of input values
    - weights: array of weights
    - activation_type: 'threshold' or 'sigmoid'
    - threshold: threshold value (theta)
    - gain: gain parameter for sigmoid

    Returns:
    - prediction: output after activation
    - net_input: raw net input value
    - raw_output: output before threshold comparison (for sigmoid confidence)
    """
    net_input = calculate_net_input(features, weights)

    if activation_type == 'threshold':
        prediction = activation_threshold(net_input, threshold)
        raw_output = net_input
    else:  # sigmoid
        raw_output = activation_sigmoid(net_input, threshold, gain)
        # For binary classification, use 0.5 as decision boundary
        prediction = np.where(raw_output >= 0.5, 1, 0)

    return prediction, net_input, raw_output


def get_activation_curve(activation_type='threshold', threshold=0, gain=1.0, num_points=200):
    """
    Generate activation function curve data for visualization

    Returns:
    - x: array of net input values
    - y: array of activation outputs
    """
    x = np.linspace(-5, 5, num_points)

    if activation_type == 'threshold':
        y = activation_threshold(x, threshold)
    else:  # sigmoid
        y = activation_sigmoid(x, threshold, gain)

    return x, y


def get_formula_latex(activation_type='threshold', gain=1.0):
    """
    Get LaTeX formula for the activation function
    """
    if activation_type == 'threshold':
        formula = r"""
\begin{aligned}
z &= \sum_{i=1}^{n} w_i x_i - \theta \\
\hat{y} &= \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
\end{aligned}
"""
    else:
        formula = r"""
\begin{{aligned}}
z &= \sum_{{i=1}}^{{n}} w_i x_i - \theta \\
y &= \frac{{1}}{{1 + e^{{-{gain} \cdot z}}}}
\end{{aligned}}
""".format(gain=gain)

    return formula


def train_perceptron(X, y, initial_weights, theta, learning_rate, epochs,
                     activation_type='threshold', gain=1.0):
    """
    Train the perceptron using online learning (row-by-row / sequential update).

    Core formulas:
      z        = sum(w_i * x_i) - θ
      y_hat    = 1 if z >= 0 else 0          (threshold)
               = sigmoid(z) >= 0.5           (sigmoid)
      error    = y - y_hat   (ε)
      Δw_i     = α * ε * x_i
      Δθ       = -α * ε
      w_new    = w_old + Δw
      θ_new    = θ_old + Δθ

    Immediate (online) update: weights and θ are updated after EVERY sample,
    and the new values are used for the very next sample.

    Parameters:
    - X              : Training data, shape (n_samples, n_features)
    - y              : Target labels, shape (n_samples,)
    - initial_weights: List/array of initial weight values
    - theta          : Initial threshold value (θ)
    - learning_rate  : Learning rate (α)
    - epochs         : Maximum number of full-pass epochs
    - activation_type: 'threshold' or 'sigmoid'
    - gain           : Gain for sigmoid function

    Returns:
    - final_weights  : Updated weights after training
    - final_theta    : Updated threshold after training
    - training_log   : List of dicts with step-by-step computation details
    """
    weights = np.array(initial_weights, dtype=float).copy()
    theta   = float(theta)
    X       = np.array(X, dtype=float)
    y       = np.array(y, dtype=float)

    n_samples  = X.shape[0]
    training_log = []
    epoch_errors = []
    row_counter  = 0

    for epoch in range(1, epochs + 1):
        epoch_has_error = False
        current_epoch_error_sum = 0

        for sample_idx in range(n_samples):
            row_counter += 1
            x_sample  = X[sample_idx]
            y_desired = y[sample_idx]

            # --- Step 1: Compute z = sum(w_i * x_i) - θ ---
            z = float(np.dot(x_sample, weights) - theta)

            # --- Step 2: Predict y_hat ---
            if activation_type == 'threshold':
                y_hat = 1.0 if z >= 0 else 0.0
            else:  # sigmoid
                y_hat = 1.0 / (1.0 + np.exp(-gain * z))  # Raw sigmoid output (no rounding)

            # --- Step 3: Error ε = y - y_hat ---
            error = float(y_desired - y_hat)
            current_epoch_error_sum += abs(error)
            if error != 0:
                epoch_has_error = True

            # --- Step 4: Compute deltas ---
            weight_deltas = learning_rate * error * x_sample   # Δw
            theta_delta   = -learning_rate * error             # Δθ

            # Save values before update
            weights_before = weights.copy()
            theta_before   = theta

            # --- Step 5: Immediate update ---
            weights = weights + weight_deltas
            theta   = theta   + theta_delta

            # --- Log entry ---
            log_entry = {
                'n'            : row_counter,
                'epoch'        : epoch,
                'sample'       : sample_idx + 1,
                'inputs'       : x_sample.tolist(),
                'desired'      : float(y_desired),
                'z'            : z,
                'predicted'    : float(y_hat),
                'error'        : float(error),
                'weights_before': weights_before.tolist(),
                'weight_deltas': weight_deltas.tolist(),
                'weights_after': weights.tolist(),
                'theta_before' : float(theta_before),
                'theta_delta'  : float(theta_delta),
                'theta_after'  : float(theta),
            }
            training_log.append(log_entry)

        epoch_errors.append(current_epoch_error_sum)
        
        # Early stopping: if the whole epoch had no errors, converged
        if not epoch_has_error:
            break

    return weights, theta, training_log, epoch_errors
