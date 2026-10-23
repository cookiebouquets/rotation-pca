# Rotation-Based Subspace Tracking for Robust Kernel PCA on Streaming Data

Machine learning models process large amounts of data, and Principal Component Analysis (PCA) is a widely used technique to reduce the dimensionality of the data and extract useful features. In practice, datasets often change over time (data drift) and/or arrive one sample at a time (streaming data), making it infeasible to process the entire dataset at once in batch mode. Real-world data also often contains nonlinear patterns, which traditional PCA cannot extract. Kernel PCA addresses this by implicitly mapping samples into a Reproducing Kernel Hilbert Space (RKHS). Raw data also often contains outliers, which can have an outsized effect on the estimated subspace unless the algorithm is made robust. However, existing online robust kernel PCA algorithms are designed to converge to a subspace that is assumed to be fixed, and gradient-descent-based updates lose their effectiveness at tracking further changes once this initial alignment is achieved. This paper introduces a rotation-based update mechanism, which updates the subspace estimate by rotating it toward each new incoming feature vector in Reproducing Kernel Hilbert Space, rather than relying on gradient descent alone. We present two complementary rotation strategies, and show that the extent of rotation can be moderated by a robust influence function to mitigate the effect of outliers. Through experiments on synthetic streaming data with a known ground-truth subspace, we show that per-sample rotations converge faster than gradient descent alone, demonstrating an effective mechanism for dynamically tracking a nonlinear subspace in streaming data.

Read the paper here: ![arxiv link](https://arxiv.org/abs/2609.15488)

## Usage

To run the experiments outlined in the paper, first copy the github repo and run 

```
uv sync
```

then, you can run any experiment in the `results` directory by running

```
uv run python -m results.<name of python script>
```

This will create an untracked `plots` directory in the project's root directory with the plots of the experiment. 

