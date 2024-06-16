import skimage
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.morphology import label
from skimage.measure import regionprops


# Read in the corrupted coins image
coins = skimage.io.imread('data/coins.png', as_gray=True)

# Use impainting to interpolate dark areas in the image
mask = coins < 0.05
coins_inpainted = inpaint.inpaint_biharmonic(coins, mask)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(coins, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(coins_inpainted, cmap='gray')
plt.axis('off')
plt.title('Impainted Image')
plt.tight_layout()
plt.savefig('figures/coins_original+inpaint.png')

# Use thresholding to segment the image
binary = coins_inpainted > 0.5

# Use Binary Closing to fill in the holes
closed = binary_closing(binary, disk(3))
closed = remove_small_holes(closed, 200)
label_image = label(closed)
regions = regionprops(label_image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Image after thresholding')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(closed, cmap='gray')
plt.title('Image after closing and removing small holes')
plt.axis('off')

# delete non-circular regions
deleted = []
for lb in range(len(regions)):
    if regions[lb].eccentricity > 0.5:
        label_image[label_image == lb+1] = 0
        deleted.append(lb)

# Delete small regions that are not coins
filtered_image = np.zeros_like(label_image)
min_area = 200

for region in regions:
    if region.area >= min_area:
        filtered_image[label_image == region.label] = region.label

# Repropagate the regions after filtering
filtered_regions = regionprops(filtered_image)

plt.subplot(1, 3, 3)
plt.imshow(filtered_image, cmap='nipy_spectral')
plt.title('Labelled regions (as highlighted) after removal' +
          "\nof small/non-circular regions")
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/coins_labelled_regions.png')


# Sort the regions in the filtered image so that the indices of the regions
# correspond to the order of the coins in the original image
# First, sort the centroids from top to bottom
centroids = [region.centroid for region in filtered_regions]
sorted_regions = [region for _, region in sorted(zip(centroids,
                                                     filtered_regions),
                                                 key=lambda x: x[0])]

# For every 6 centroids, sort them from left to right (row-wise sort)
for i in range(0, len(sorted_regions), 6):
    sorted_regions[i:i+6] = sorted(sorted_regions[i:i+6],
                                   key=lambda x: x.centroid[1])

# Plot the coins with the specified regions (first coin in first row,
# second coin in second row, etc.)
segmented_coins = coins*((label_image == sorted_regions[0].label) +
                         (label_image == sorted_regions[7].label) +
                         (label_image == sorted_regions[14].label) +
                         (label_image == sorted_regions[21].label))

plt.figure(figsize=(10, 5))
plt.imshow(segmented_coins, cmap='gray')
plt.axis('off')
