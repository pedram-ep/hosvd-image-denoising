import numpy as np

def aggregate_patches(denoised_patches, locations, image_shape, patch_size):
    accumulator = np.zeros(image_shape)
    count = np.zeros(image_shape)

    for group, locs in zip(denoised_patches, locations):
        for k in range(group.shape[-1]):
            i, j = locs[k]
            patch = group[:, :, k]
            accumulator[i:i+patch_size, j:j+patch_size] += patch
            count[i:i+patch_size, j:j+patch_size] += 1

    count[count == 0] = 1
    return accumulator / count