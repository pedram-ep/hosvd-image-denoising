import matplotlib.pyplot as plt
from .denoiser    import hosvd_denoise
from .metrics     import compute_image_metrics

def plot_comparison(idx: int, original_images):
    orig_img = original_images[idx]
    noisy_images = []
    denoised_images = []
    metrics_texts_noisy = []
    metrics_texts_denoised = []

    # gather images and metrics
    for folder, sigma in zip(folders[1:], sigmas):
        noisy_img = images[folder][idx]
        denoised_img = hosvd_denoise(
            noisy_img, sigma=sigma,
            patch_size=8, search_radius=20,
            max_patches=30, stride=4
        )
        metrics = compute_image_metrics(
            original=orig_img, noisy=noisy_img,
            denoised=denoised_img,
            data_range=orig_img.max() - orig_img.min()
        )
        noisy_images.append(noisy_img)
        denoised_images.append(denoised_img)

        fmt = lambda m: (
            f"MSE: {m['MSE']:.2f} | RMSE: {m['RMSE']:.2f} | "
            f"MAE: {m['MAE']:.2f} | PSNR: {m['PSNR']:.2f} | "
            f"SSIM: {m['SSIM']:.3f} | NRMSE: {m['NRMSE']:.3f} | UQI: {m['UQI']:.3f}"
        )
        metrics_texts_noisy.append(fmt(metrics['noisy_vs_original']))
        metrics_texts_denoised.append(fmt(metrics['denoised_vs_original']))

    # 11 rows x 2 columns
    fig, axes = plt.subplots(11, 2, figsize=(20, 40))
    fig.suptitle(f"Image Index {idx:02d} — Original, Noisy vs Denoised", fontsize=24)

    # Row 0: original
    axes[0, 0].imshow(orig_img, cmap='gray')
    axes[0, 0].set_title("Original", fontsize=18)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    # Rows 1–10: for each sigma, two rows each
    for i, sigma in enumerate(sigmas):
        row_img = 1 + 2 * i
        row_txt = row_img + 1

        # image row
        axes[row_img, 0].imshow(noisy_images[i], cmap='gray')
        axes[row_img, 0].set_title(f"Noisy (σ={sigma})", fontsize=16)
        axes[row_img, 0].axis('off')

        axes[row_img, 1].imshow(denoised_images[i], cmap='gray')
        axes[row_img, 1].set_title(f"Denoised (σ={sigma})", fontsize=16)
        axes[row_img, 1].axis('off')

        # metrics row
        axes[row_txt, 0].text(
            0.5, 0.5, metrics_texts_noisy[i],
            fontsize=12, ha='center', va='center', wrap=True
        )
        axes[row_txt, 0].axis('off')

        axes[row_txt, 1].text(
            0.5, 0.5, metrics_texts_denoised[i],
            fontsize=12, ha='center', va='center', wrap=True
        )
        axes[row_txt, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
