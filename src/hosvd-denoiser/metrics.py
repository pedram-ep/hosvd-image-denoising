import numpy as np
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    normalized_root_mse
)
from sklearn.metrics import mean_squared_error

def compute_image_metrics(original, noisy, denoised, data_range=None):
    if data_range is None:
        data_range = np.ptp(original)

    def _uqi(x, y):
        cov = np.cov(x.flatten(), y.flatten())[0, 1]
        mean_x, mean_y = np.mean(x), np.mean(y)
        var_x, var_y = np.var(x), np.var(y)
        return (4 * cov * mean_x * mean_y) / ((var_x + var_y) * (mean_x**2 + mean_y**2) + 1e-10)

    def _compute_metrics_pair(ref, target, prefix):
        metrics = {}
        metrics['MSE'] = mean_squared_error(ref, target)
        metrics['RMSE'] = np.sqrt(metrics['MSE'])
        metrics['MAE'] = np.mean(np.abs(ref - target))
        metrics['PSNR'] = peak_signal_noise_ratio(ref, target, data_range=data_range)
        metrics['SSIM'] = structural_similarity(ref, target, data_range=data_range, win_size=7)
        metrics['NRMSE'] = normalized_root_mse(ref, target, normalization='mean')
        metrics['UQI'] = _uqi(ref, target)
        return metrics

    results = {
        'noisy_vs_original': _compute_metrics_pair(original, noisy, 'noisy'),
        'denoised_vs_original': _compute_metrics_pair(original, denoised, 'denoised')
    }

    return results