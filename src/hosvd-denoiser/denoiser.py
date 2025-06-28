import tensorly as tl
from tensorly.decomposition import tucker
from .patch_grouping import create_patch_groups
from .aggregation import aggregate_patches

def hosvd_denoise(noisy_image, sigma=25, patch_size=8, search_radius=20, max_patches=30, stride=4):
    groups, locations = create_patch_groups(
        noisy_image, patch_size, sigma, search_radius, max_patches, stride
    )

    denoised_groups = []
    total_groups = len(groups)

    for idx, group in enumerate(groups):
        # # Progress tracking
        # if idx % 1000 == 0:
        #     print(f"Processing group {idx}/{total_groups}")

        K = group.shape[-1]

        if K < 2:
            denoised_groups.append(group)
            continue

        core, factors = tucker(
            group,
            rank=[min(dim, max_rank) for dim, max_rank in zip(group.shape, [patch_size, patch_size, max_patches])],
            init='svd'
        )

        threshold = sigma * np.sqrt(2 * np.log(patch_size**2 * K))
        core_thresholded = np.where(np.abs(core) < threshold, 0, core)

        denoised_group = tl.tucker_to_tensor((core_thresholded, factors))
        denoised_groups.append(denoised_group)

    return aggregate_patches(denoised_groups, locations, noisy_image.shape, patch_size)