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
a_opt, b_opt = result.x

print("The fitted coefficients for L1 minimisation with noisy data \
       are a = {:.3f} and b = {:.3f}".format(a_opt, b_opt))

# Plot the data and the fitted line
plt.scatter(x, y_line, label='Data')
plt.plot(x, a_opt * x + b_opt, color='red', label='LAD fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L1 fit to the noisy data')
plt.savefig('L1_minimisation_noisy_data.png')
plt.close()

# Fit the line to the outlier data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L1_objective_function, initial_guess,
                                 args=(x, y_outlier_line))

# Extract the optimal parameters
a_opt, b_opt = result.x

print("The fitted coefficients for L1 minimisation with outlier data \
         are a = {:.3f} and b = {:.3f}".format(a_opt, b_opt))

# Plot the data and the fitted line
plt.scatter(x, y_outlier_line, label='Data')
plt.plot(x, a_opt * x + b_opt, color='red', label='LAD fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L1 fit to the outlier data')
plt.savefig('L1_minimisation_outlier_data.png')
plt.close()

# Fit the line to the data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L2_objective_function, initial_guess,
                                 args=(x, y_line))

# Extract the optimal parameters
a_opt, b_opt = result.x

print("The fitted coefficients for L2 minimisation with noisy data \
            are a = {:.3f} and b = {:.3f}".format(a_opt, b_opt))

# Plot the data and the fitted line
plt.scatter(x, y_line, label='Data')
plt.plot(x, a_opt * x + b_opt, color='red', label='L2 fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L2 fit to the noisy data')
plt.savefig('L2_minimisation_noisy_data.png')
plt.close()

# Fit the line to the outlier data
initial_guess = [1, 1]
result = scipy.optimize.minimize(L2_objective_function, initial_guess,
                                 args=(x, y_outlier_line))

# Extract the optimal parameters
a_opt, b_opt = result.x

print("The fitted coefficients for L2 minimisation with outlier data \
            are a = {:.3f} and b = {:.3f}".format(a_opt, b_opt))

# Plot the data and the fitted line
plt.scatter(x, y_outlier_line, label='Data')
plt.plot(x, a_opt * x + b_opt, color='red', label='L2 fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('L2 fit to the outlier data')
plt.savefig('L2_minimisation_outlier_data.png')
plt.close()
