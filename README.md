# Non_uniform_sampling
A complete, reproducible pipeline for screening mammography that follows a patch → heatmap → non-uniform warp → whole-image strategy.
Built around VinDr-Mammo, it (1) preprocesses images, (2) creates S and s10 patch datasets, (3) trains a patch classifier, (4) produces lesion-probability heatmaps, and (5) performs saliency-guided warping controlled by scale and FWHM to emphasize suspicious regions for the whole-image classifier.
