import time
import numpy as np
from .offsets import generate_spiral_offsets

def create_patch_groups(image, patch_size, sigma, search_radius=20, max_patches=30, stride=4):
    tau_d = 3 * (sigma ** 2) * (patch_size ** 2)
    H, W = image.shape

    all_patches = np.lib.stride_tricks.sliding_window_view(image, (patch_size, patch_size))
    Hp, Wp = all_patches.shape[:2]

    flat_patches = all_patches.reshape(Hp, Wp, -1)
    patch_norms = np.einsum('ijk,ijk->ij', flat_patches, flat_patches)

    spiral_offsets = generate_spiral_offsets(search_radius)

    groups = []
    locations = []
    total_refs = ((Hp - 1) // stride + 1) * ((Wp - 1) // stride + 1)
    processed = 0
    start_time = time.time()

    for i in range(0, Hp, stride):
        for j in range(0, Wp, stride):
            # # Progress tracking
            # processed += 1
            # if processed % 10000 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"Processed {processed}/{total_refs} reference patches - {elapsed:.2f}s")

            group = [all_patches[i, j]]
            locs = [(i, j)]

            for ring in spiral_offsets:
                for dx, dy in ring:
                    x, y = i + dx, j + dy
                    if 0 <= x < Hp and 0 <= y < Wp:
                        dist = patch_norms[i, j] + patch_norms[x, y] - 2 * np.dot(flat_patches[i, j], flat_patches[x, y])

                        if dist < tau_d:
                            group.append(all_patches[x, y])
                            locs.append((x, y))
                            if len(group) >= max_patches:
                                break
                if len(group) >= max_patches:
                    break

            groups.append(np.stack(group, axis=-1))
            locations.append(locs)

    return groups, locations