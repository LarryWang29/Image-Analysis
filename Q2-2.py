import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from helper import ComplexSoftThresh
from copy import deepcopy

# Fix a seed for reproducibility
np.random.seed(1029)

# Generate a length 128 vector of 0's
x = np.zeros(100)

# Randomly set 10 indices to uniform samples from [0,4], which will
# be the true signal
x[np.random.choice(100, 10, replace=False)] = 4 * np.random.rand(10)

# Add Gaussian noise with std dev = 0.05
x_noisy = x + 0.05 * np.random.randn(100)

# Visualize the true signal and the noisy signal
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.stem(x)
plt.title('Original Signal')
plt.subplot(2, 1, 2)
plt.stem(x_noisy)
plt.title('Noisy Signal')
plt.tight_layout()
plt.savefig('figures/original+noisy_signals.png')

# Define the 1D FFT functions (Adapted from fft functions in helper.py)
def fftc(x):
    """
    This function computes the centered 1D FFT of the input signal x, used
    for the part 2 of Question 2.

    Parameters
    ----------
    x : numpy.ndarray
        The input signal to be transformed.

    Returns
    -------
    numpy.ndarray
        The centered 1D FFT of the input signal.
    """
    return 1 / np.sqrt(np.prod(x.shape)) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))

def ifftc(y):
    """
    This function computes the inverse centered 1D FFT of the input signal y,
    used for the part 2 of Question 2.

    Parameters
    ----------
    y : numpy.ndarray
        The input signal to be transformed.
    
    Returns
    -------
    numpy.ndarray
        The inverse centered 1D FFT of the input signal.
    """
    return np.sqrt(np.prod(y.shape)) * np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(y)))

# Pad the noisy signal with zeros to length 128 so that it's a power of 2
x_noisy_padded = np.pad(x_noisy, (0, 28), mode='constant')
y = fftc(x_noisy_padded)

# Take 96 samples at random to zero out for random sampling
idx_random = np.random.choice(128, 32, replace=False)

# Also take 32 equidistant samples for uniform sampling
idx_uniform = np.arange(0, 128, 4)

# Obtain the undersampled fourier coefficients using different sampling schemes
random_fourier = np.zeros(128, dtype=complex)
uniform_fourier = np.zeros(128, dtype=complex)

random_fourier[idx_random] = y[idx_random]
uniform_fourier[idx_uniform] = y[idx_uniform]

plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.scatter(idx_random, y[idx_random], c='r')
plt.title('Randomly Sampled Fourier Coefficients')
plt.subplot(2, 1, 2)
plt.plot(y)
plt.scatter(idx_uniform, y[idx_uniform], c='r')
plt.title('Equidistantly Sampled Fourier Coefficients')
plt.tight_layout()
plt.savefig('figures/fourier_samples.png')

# Reconstruct the signals using the inverse FFT
x_random = ifftc(random_fourier)
x_uniform = ifftc(uniform_fourier)

# Multiply by 4 to account for the fact that we only took 1/4 of the samples
x_random *= 4
x_uniform *= 4

# Take the first 100 samples since the last 28 are just padding
x_random = x_random[:100]
x_uniform = x_uniform[:100]

# Visualise these reconstructions
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.stem(np.real(x_random), linefmt='b--')
plt.stem(np.imag(x_random), linefmt='r--')
plt.stem(x, linefmt='g--')
plt.title('Reconstruction from random sampling')
legend_elements = [Line2D([0], [0], color='b', lw=2, linestyle='--', label='Real Components'),
                   Line2D([0], [0], color='r', lw=2, linestyle='--', label='Imaginary Components'),
                   Line2D([0], [0], color='g', lw=2, linestyle='--', label='Original Signal')]
plt.legend(handles=legend_elements)
plt.subplot(2, 1, 2)
plt.stem(np.real(x_uniform), linefmt='b--')
plt.stem(np.imag(x_uniform), linefmt='r--')
plt.stem(x, linefmt='g--')
plt.title('Reconstruction from uniform sampling')
legend_elements = [Line2D([0], [0], color='b', lw=2, linestyle='--', label='Real Components'),
                   Line2D([0], [0], color='r', lw=2, linestyle='--', label='Imaginary Components'),
                   Line2D([0], [0], color='g', lw=2, linestyle='--', label='Original Signal')]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig('figures/reconstructions.png')

# Use Complex thresholding to recover the signal for both cases
def iterative_solution(y, niter=100, L=0.1):
    # Initialize x_hat to be the reconstruction from the inverse FFT
    X_hat = deepcopy(y)

    # Find the indices where the original signal is not 0 for consistency
    # checking
    idx = np.argwhere(y != 0)

    for _ in range(niter):
        # Take the inverse FFT
        s_approx = ifftc(X_hat)
        # Apply the soft-thresholding function
        s_hat = ComplexSoftThresh(s_approx, L)
        # Take the FFT
        X_hat_new = fftc(s_hat)
        # If the original y is not 0, then reset the values of X_hat_new to be the original y
        # to ensure consistency
        X_hat_new[idx] = y[idx]
        X_hat = X_hat_new
    return s_hat[:100]

lambdas = [0.01, 0.05, 0.1]
plt.figure(figsize=(20, 20))
current_plot = 1

for lam in lambdas:
    # Calculate the solutions for this lambda
    x_solution_uniform = iterative_solution(uniform_fourier, L=lam)
    x_solution_random = iterative_solution(random_fourier, L=lam)

    # Plot solution from random sampling and original signal
    plt.subplot(3, 2, current_plot)
    plt.stem(x_solution_random, linefmt='b--')
    plt.stem(x, linefmt='g--')
    plt.title(rf'Comparison of randomly undersampling solution and original signal, $\lambda={lam}, $' +
              f'Error={np.linalg.norm(x - x_solution_random).round(3)}')
    legend_elements = [Line2D([0], [0], color='b', lw=2, linestyle='--', label='Iterative Solution'),
                    Line2D([0], [0], color='g', lw=2, linestyle='--', label='Original Signal')]
    plt.legend(handles=legend_elements)
    plt.subplot(3, 2, current_plot + 1)

    # Plot solution from uniform sampling and original signal
    plt.stem(x_solution_uniform, linefmt='b--')
    plt.stem(x, linefmt='g--')
    plt.title(rf'Comparison of uniform undersampling solution and original signal, $\lambda={lam}, $' +
              f'Error={np.linalg.norm(x - x_solution_uniform).round(3)}')
    legend_elements = [Line2D([0], [0], color='b', lw=2, linestyle='--', label='Iterative Solution'),
                    Line2D([0], [0], color='g', lw=2, linestyle='--', label='Original Signal')]
    plt.legend(handles=legend_elements)
    current_plot += 2

plt.tight_layout()
plt.savefig('figures/solution_comparison.png')
