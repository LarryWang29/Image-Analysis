import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_closing, binary_opening, disk, \
remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops

ct = skimage.io.imread('data/CT.png', as_gray=True)

# Convert the image to 8-bit for easier binning during Otsu thresholding
ct = (ct * 255).astype(np.uint8)

# Custom Otsu thresholding function
def otsu_threshold(image):
    """
    This is a custom implementation of Otsu's thresholding algorithm. It
    computes the optimal threshold value for binarising an image based on the
    intra-class variance of pixel intensities. This is used for the
    lung segmentation section of Question 1.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be thresholded.
    
    Returns
    -------
    int
        The optimal threshold value for binarising the image.
    """
    # Compute histogram
    hist = np.histogram(image, bins=256, range=(0, 256), density=True)[0]
    best_thresh = np.inf
    best_bin = 0
    for bins in range(len(hist)):
        # Compute class probabilities
        w0 = np.sum(hist[:bins])
        w1 = np.sum(hist[bins:])
        # Compute class means and variances
        if w0 != 0:
            mu0 = np.sum([i * hist[i] for i in range(bins)]) / w0
            sigma0 = np.sum([(i - mu0) ** 2 * hist[i] for i in range(bins)]) / w0
        else:
            mu0 = 0
            sigma0 = 0
        if w1 != 0:
            mu1 = np.sum([i * hist[i] for i in range(bins, 256)]) / w1
            sigma1 = np.sum([(i - mu1) ** 2 * hist[i] for i in range(bins, 256)]) / w1
        else:
            mu1 = 0
            sigma1 = 0
        
        # Compute intra-class variance
        sigma_w = w0 * sigma0 + w1 * sigma1
        # Update threshold if intra-class variance is lower
        if sigma_w < best_thresh:
            best_thresh = sigma_w
            best_bin = bins

    # Account for the fact that the threshold is the lower bound of the higher
    # class
    return best_bin - 1

plt.figure(figsize=(8, 5))
plt.hist(ct.ravel(), bins=256, range=(0, 256), density=True)
plt.xlabel('Pixel intensity')
plt.ylabel('Frequency')
plt.title('Histogram of pixel intensities')
plt.tight_layout()
plt.savefig("figures/CT_histogram.png")

ct_thresh = ct <= otsu_threshold(ct)

# Apply morphological operation to the thresholded image to smooth out internal
# structures
footprint = disk(4)
ct_opened = binary_opening(ct_thresh, footprint)
ct_closed = binary_closing(ct_opened, footprint)
plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1)
plt.imshow(ct_thresh, cmap='gray')
plt.axis('off')
plt.title('Thresholded image')
plt.subplot(1, 3, 2)
plt.imshow(ct_closed, cmap='gray')
plt.axis('off')
plt.title('Image after closing and opening')

ct_closed = remove_small_holes(ct_closed, 1000)
ct_closed = remove_small_objects(ct_closed, 1000)
plt.subplot(1, 3, 3)
plt.imshow(ct_closed, cmap='gray')
plt.axis('off')
plt.title('Image after removing small holes and small objects')
plt.tight_layout()
plt.savefig("figures/CT_segmented_steps.png")

# Use label to segment out the lungs
ct_labels = label(ct_closed)
regions = regionprops(ct_labels)

# Only keep the second and third largest regions
lung_labels = np.zeros_like(ct_labels)

# Sort the regions by area from smallest to largest
sorted_regions = sorted(regions, key=lambda x: x.area)

# Visualise the two smallest regions (the lungs)
lung_labels[ct_labels == sorted_regions[0].label] = 1
lung_labels[ct_labels == sorted_regions[1].label] = 1

# Display the segmented lungs
plt.figure(figsize=(5, 4))
plt.imshow(ct, cmap='gray')
plt.imshow(lung_labels, cmap='inferno', alpha=0.3)
plt.axis('off')
plt.title('Segmented lungs overlaid')
plt.tight_layout()
plt.savefig("figures/CT_segmented_lungs.png")
