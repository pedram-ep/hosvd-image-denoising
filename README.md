# Image Denoising via Higher Order SVD (HOSVD)

This repository contains a Python implementation of patch-based image denoising using Higher Order Singular Value Decomposition (HOSVD), based on the method introduced in:

> **"Image Denoising using the Higher Order Singular Value Decomposition"**  
> Ajit Rajwade, Anand Rangarajan, and Arunava Banerjee.  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2013.

<p align="center">
  <img src="results/figures/image-00-results-cropped.png?raw=true" alt="Denoising example" width="600">
</p>

---

##  Methodology
The algorithm works by:
1. Partitioning images into overlapping 8×8 patches
2. Grouping similar patches using spiral window search
3. Constructing 3D tensors for each patch group
4. Applying HOSVD decomposition
5. Thresholding core tensor values (hard threshold at τ = σ√(s²K))
6. Reconstructing denoised patches
7. Aggregating patches with averaging

*Note: Implementation is optimized for grayscale images and omits the Wiener filtering step from the original paper*

##  Evaluation Metrics
We used seven metrics to evaluate denoising performance on the CBSD68 dataset:

| Metric | Formula | Description | Ideal Value |
|--------|---------|-------------|-------------|
| **MSE** | $\frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}[I(i,j)-K(i,j)]^2$ | Mean Squared Error | Lower is better |
| **RMSE** | $\sqrt{\text{MSE}}$ | Root Mean Squared Error | Lower is better |
| **MAE** | $\frac{1}{MN}\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}\|I(i,j)-K(i,j)\|$ | Mean Absolute Error | Lower is better |
| **PSNR** | $10 \cdot \log_{10}\left(\frac{L^2_{\max}}{\text{MSE}}\right)$ | Peak Signal-to-Noise Ratio | Higher is better |
| **SSIM** | $\frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$ | Structural Similarity Index | Closer to 1 |
| **NRMSE** | $\frac{\sqrt{\frac{1}{N}\sum_{i=1}^N(x_i - y_i)^2}}{\text{median}(\|x\|)}$ | Normalized RMSE | Lower is better |
| **UQI** | $\frac{4\sigma_{xy}\bar{x}\bar{y}}{(\sigma_x^2 + \sigma_y^2)(\bar{x}^2 + \bar{y}^2)}$ | Universal Quality Index | Closer to 1 |

## Repository Structure
```
├── notebooks/              # Jupyter notebooks for workflow
├── src/                    # Core implementation
│   ├── denoiser.py
│   ├── patch_grouping.py
│   ├── aggregation.py
│   ├── offsets.py
│   ├── metrics.py
│   └── visualization.py
├── results/
│   ├── figures/            # Visual results
│   ├── final_results.csv   # Complete metric data
│   └── mean_by_sigma.csv   # Averaged results by noise level
└── requirements.txt
```

---

## Installation
```bash
git clone https://github.com/pedram-ep/hosvd-image-denoising.git
cd image-denoising-hosvd

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## References

1. Rajwade, A., Rangarajan, A., & Banerjee, A. (2013). "*Image Denoising using the Higher Order Singular Value Decomposition*".
2. Feschet, F. (2019). “*Implementation of a denoising algorithm based on High‑Order Singular Value Decomposition of tensors*”
3. Guo, J., Chen, H., Shen, Z., and Wang, Z. (2022). “*Image denoising based on global image similar patches searching and HOSVD to patches tensor*”
4. Wang, Z., and Bovik, A. C. (2002). “*A universal image quality index*”
5. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P., “*Image quality assessment: From error visibility to structural similarity*”
6. CBSD68‑dataset: Color BSD68 dataset for image denoising benchmarks. Available at: [GitHub](https://github.com/clausmichele/CBSD68-dataset)