import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import binary_closing, remove_small_objects, disk
from skimage.color import rgb2lab
from sklearn.cluster import KMeans


def find_purple_cluster(kmeans_cluster):
    """
    This function finds the cluster closest to the purple lab color in the
    KMeans cluster object. This is used for the flower segmentation section
    of Question 1.

    Parameters
    ----------
    kmeans_cluster : sklearn.cluster.KMeans
        The KMeans cluster object containing the cluster centers.
    
    Returns
    -------
    purple_cluster : int
        The index of the cluster closest to the purple lab color.
    """
    # Purple color in LAB space 
    # (from https://convertingcolors.com/cielab-color-45.36_78.75_-77.41.html)
    purple_lab = np.array([45.36, 78.75, -77.41])
    distances = np.sum((kmeans_cluster.cluster_centers_ - purple_lab) ** 2,
                       axis=1)

    # Find the cluster closest to the purple lab color
    purple_cluster = np.argmin(distances)
    return purple_cluster


# Read the images
flowers = skimage.io.imread('data/noisy_flower.jpg')[:, :, :3]

# Denoise using Total Variation filter
flowers_denoised = denoise_tv_chambolle(flowers, weight=0.1)

# Plot both the originial and the denoised images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(flowers)

# Hide the axes
plt.axis('off')
plt.title('Original noisy image')

plt.subplot(1, 2, 2)
plt.imshow(flowers_denoised)

# Hide the axes
plt.axis('off')
plt.title('Denoised image')
plt.tight_layout()
plt.savefig('figures/denoised_comparison.png')

# Convert from RGB to LAB color space for better clustering
flowers_lab = rgb2lab(flowers_denoised)
flowers_noisy_lab = rgb2lab(flowers)

# Perform KMeans clustering on the denoised and noisy images
kmeans_denoised = KMeans(n_clusters=5,
                         random_state=0,
                         n_init=25).fit(flowers_lab.reshape(-1, 3))

kmeans_noisy = KMeans(n_clusters=5,
                      random_state=0,
                      n_init=25).fit(flowers_noisy_lab.reshape(-1, 3))

# Find the cluster closest to the purple lab color
purple_cluster_denoised = find_purple_cluster(kmeans_denoised)
purple_cluster_noisy = find_purple_cluster(kmeans_noisy)

# Reshape the labels to the original image shape
denoised_labels = kmeans_denoised.labels_.reshape(666, 1000)
noisy_labels = kmeans_noisy.labels_.reshape(666, 1000)

# Plot the noisy and denoised images with the purple cluster highlighted
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(noisy_labels == purple_cluster_noisy, cmap='gray')

# Hide the axes
plt.axis('off')
plt.title('Purple cluster in noisy image')

plt.subplot(1, 2, 2)
plt.imshow(denoised_labels == purple_cluster_denoised, cmap='gray')

# Hide the axes
plt.axis('off')
plt.title('Purple cluster in denoised image')
plt.tight_layout()
plt.savefig('figures/purple_cluster.png')

# Further denoise the cluster by performing binary closing followed by
# binary opening
purple_mask_closed = binary_closing(denoised_labels == purple_cluster_denoised,
                                    disk(2))
purple_mask_opened = remove_small_objects(purple_mask_closed, min_size=150)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(purple_mask_closed, cmap='gray')
plt.axis('off')
plt.title('Purple cluster after binary closing')
plt.subplot(1, 2, 2)
plt.imshow(purple_mask_opened, cmap='gray')
plt.axis('off')
plt.title('Purple cluster after binary closing and removal of small objects')
plt.tight_layout()
plt.savefig('figures/morphological_operations.png')

# Plot the original image with the purple cluster highlighted
plt.figure(figsize=(5, 4))
plt.imshow(flowers)
# Highlight the purple cluster
plt.imshow(purple_mask_opened, cmap='RdPu', alpha=0.5)
plt.axis('off')
plt.title('Purple cluster in original image')
plt.tight_layout()
plt.savefig('figures/purple_cluster_original.png')
