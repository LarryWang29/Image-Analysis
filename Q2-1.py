import numpy as np
import scipy
import matplotlib.pyplot as plt

# Read in the data from y_line and y_outlier_line
y_line = np.loadtxt('data/y_line.txt', dtype=float)
y_outlier_line = np.loadtxt('data/y_outlier_line.txt', dtype=float)

x = np.arange(0, 20, 1)


def L1_objective_function(m, x, y):
    a, b = m
    return np.sum(np.abs(y - (a * x + b)))


def L2_objective_function(m, x, y):
    a, b = m
    return np.sum((y - (a * x + b))**2)


# Fit the line to the data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L1_objective_function, initial_guess,
                                 args=(x, y_line))

# Extract the optimal parameters
a_opt_l1_noisy, b_opt_l1_noisy = result.x

print("The fitted coefficients for L1 minimisation with noisy data \
       are a = {:.3f} and b = {:.3f}".format(a_opt_l1_noisy, b_opt_l1_noisy))


# Fit the line to the outlier data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L1_objective_function, initial_guess,
                                 args=(x, y_outlier_line))

# Extract the optimal parameters
a_opt_l1_outlier, b_opt_l1_outlier = result.x

print("The fitted coefficients for L1 minimisation with outlier data \
         are a = {:.3f} and b = {:.3f}".format(a_opt_l1_outlier,
                                               b_opt_l1_outlier))


# Fit the line to the data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L2_objective_function, initial_guess,
                                 args=(x, y_line))

# Extract the optimal parameters
a_opt_l2_noisy, b_opt_l2_noisy = result.x

print("The fitted coefficients for L2 minimisation with noisy data \
            are a = {:.3f} and b = {:.3f}".format(a_opt_l2_noisy,
                                                  b_opt_l2_noisy))


# Fit the line to the outlier data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L2_objective_function, initial_guess,
                                 args=(x, y_outlier_line))

# Extract the optimal parameters
a_opt_l2_outlier, b_opt_l2_outlier = result.x

print("The fitted coefficients for L2 minimisation with outlier data \
            are a = {:.3f} and b = {:.3f}".format(a_opt_l2_outlier,
                                                  b_opt_l2_outlier))

# Plot the data and the fitted line
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x, y_line, label='Data')
plt.plot(x, a_opt_l2_noisy * x + b_opt_l2_noisy, color='green',
         label='Fitted L2 line')
plt.plot(x, a_opt_l1_noisy * x + b_opt_l1_noisy, color='red',
         label='Fitted L1 line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L1 and L2 fit to noisy data')
plt.subplot(1, 2, 2)
plt.scatter(x, y_outlier_line, label='Data')
plt.plot(x, a_opt_l1_outlier * x + b_opt_l1_outlier, color='red',
         label='Fitted L1 line')
plt.plot(x, a_opt_l2_outlier * x + b_opt_l2_outlier, color='green',
         label='Fitted L2 line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L1 and L2 fit to noisy data + outlier')
plt.tight_layout()
plt.savefig('figures/Q2-1.png')
plt.close()
