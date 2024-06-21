import numpy as np
import matplotlib.pyplot as plt
import skimage
import helper
from skimage.metrics import peak_signal_noise_ratio as psnr, \
structural_similarity as ssim

# Read in the jpeg image
img = skimage.io.imread('data/river_side.jpeg', as_gray=True)

# find the bounding box of the image without the white space
# Get the indices of the non-white pixels
non_white_pixels = np.where(img < 1)

# Get the bounding box of the non-white pixels
min_x = np.min(non_white_pixels[0])
max_x = np.max(non_white_pixels[0]) + 1
min_y = np.min(non_white_pixels[1])
max_y = np.max(non_white_pixels[1]) + 1

# Pad the x bound values by 4 on each side to ensure the dimensions
# are multiples of 16
min_x -= 4
max_x += 4

# Crop the image to the bounding box
img = img[min_x: max_x, min_y: max_y]

plt.figure(figsize=(10, 7.5))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/cropped_riverside.png')

# Compute the 2D wavelet transform of the image
Wim = helper.dwt2(img)

# Visualise the coefficients in the wavelet domain
plt.figure(figsize=(10, 7.5))
plt.imshow(np.abs(Wim) > 0.01, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/wavelet_riverside.png')

# Reconstruct the image from the wavelet coefficients
img_reconstructed = helper.idwt2(Wim)

# Plot the difference image
plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(img_reconstructed, cmap='gray')
plt.axis('off')
plt.title('Reconstructed Image')
plt.subplot(1, 3, 3)
plt.imshow(np.abs(img - img_reconstructed), cmap='gray')
plt.axis('off')
plt.title('Difference Image')
plt.colorbar(fraction=0.05, pad=0.04)
plt.tight_layout()
plt.savefig('figures/diff_riverside.png')

Wim = helper.dwt2(img)

# Create a comparative plot to show the wavelet coefficients after retaining
# only the 15% largest coefficients
plt.figure(figsize=(10, 5))
thresh_15 = np.percentile(np.abs(Wim).flatten(), 100 - 0.15 * 100)
Wim_thresh = Wim * (np.abs(Wim) > thresh_15)
plt.subplot(1, 2, 1)
plt.imshow(np.abs(Wim) > 0.01, cmap='gray')
plt.axis('off')
plt.title('Wavelet coefficients')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(Wim_thresh) > 0.01, cmap='gray')
plt.axis('off')
plt.title('Wavelet coefficients with 15% threshold')
plt.tight_layout()
plt.savefig('figures/wavelet_coeffs_thresholded.png')

# Only keep the 20% largest coefficients
thresholds = [0.2, 0.1, 0.05, 0.025]
thresh_20 = np.percentile(np.abs(Wim).flatten(), 100 - thresholds[0] * 100)
thresh_10 = np.percentile(np.abs(Wim).flatten(), 100 - thresholds[1] * 100)
thresh_05 = np.percentile(np.abs(Wim).flatten(), 100 - thresholds[2] * 100)
thresh_025 = np.percentile(np.abs(Wim).flatten(), 100 - thresholds[3] * 100)

# Plot a histogram of the wavelet coefficients
plt.figure(figsize=(8, 5))

# Fix the range of the histogram to be between 0 and 1
plt.hist(np.abs(Wim).flatten(), bins=100, range=(0, 1), density=True)
plt.axvline(thresh_20, color='red', linestyle='--', label='20% Threshold')
plt.axvline(thresh_10, color='green', linestyle='--', label='10% Threshold')
plt.axvline(thresh_05, color='purple', linestyle='--', label='5% Threshold')
plt.axvline(thresh_025, color='orange', linestyle='--', label='2.5% Threshold')
plt.xlabel('Wavelet coefficient')
plt.ylabel('Frequency')
plt.title('Histogram of wavelet coefficients')
plt.legend()
plt.tight_layout()
plt.savefig('figures/wavelet_coeffs_histogram.png')


# Now reconstruct the image using the thresholded coefficients
plt.figure(figsize=(10, 15))
threshold_values = [thresh_20, thresh_10, thresh_05, thresh_025]
for i, value in enumerate(threshold_values):
    Wim_thresh = Wim * (np.abs(Wim) > value)
    img_reconstructed = helper.idwt2(Wim_thresh)

    # Compute the MSE, PSNR and SSIM between the original and reconstructed images
    mse = np.mean((img - img_reconstructed) ** 2)
    psnr_value = psnr(img, img_reconstructed, data_range=1)
    ssim_value = ssim(img, img_reconstructed, data_range=1)

    print(f'Threshold: {thresholds[i] * 100}%')
    print(f'MSE: {mse}')
    print(f'PSNR: {psnr_value}')
    print(f'SSIM: {ssim_value}')

    plt.subplot(4, 2, 2*i + 1)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.axis('off')
    plt.title(f'Reconstructed Image with {100 * thresholds[i]}% threshold')
    plt.subplot(4, 2, 2*i + 2)
    plt.imshow(np.abs(img - img_reconstructed), cmap='gray',
               vmin=0, vmax=0.1)
    plt.axis('off')
    plt.title(f'Difference Image with {100 * thresholds[i]}% threshold')
    # plt.colorbar(fraction=0.05, pad=0.04)
plt.tight_layout()
plt.savefig(f'figures/reconstructed_riverside_different_thresholds.png')
