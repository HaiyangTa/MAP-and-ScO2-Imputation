# Conventional and Deep Learning Methods for MAP and ScO₂ Imputation

This repository provides implementations of conventional and deep learning-based methods for imputing missing data in mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) recordings.

### Introduction

Mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) are important physiological signals for assessing cerebral perfusion, oxygenation, and cerebrovascular autoregulation following pediatric cardiac arrest. However, clinical recordings frequently contain missing data due to sensor disconnections, patient motion, and equipment-related interruptions, potentially compromising downstream physiological analyses. Although conventional and deep learning-based imputation methods have been widely investigated, their performance on higher-frequency physiological recordings and their impact on downstream analyses remain insufficiently explored.

This study systematically evaluates conventional and deep learning-based imputation methods for reconstructing missing MAP and ScO₂ data sampled at 0.2 Hz. Specifically, we compare their reconstruction accuracy, investigate whether they achieve stringent mean absolute error (MAE) thresholds of less than 2 mmHg for MAP and 2 percentage points for ScO₂, and assess their ability to preserve the dynamic physiological relationship between MAP and ScO₂ through wavelet transform coherence (WTC) analysis.

### Data Structure

The dataset consists of mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) recordings from 64 pediatric patients who experienced cardiac arrest at Children's Medical Center Dallas between 2019 and 2024. MAP was measured using invasive arterial monitoring, while ScO₂ was recorded using near-infrared spectroscopy (NIRS). Both signals were collected concurrently at a sampling frequency of 0.2 Hz (one sample every 5 seconds).

The original dataset contains 5,794.46 hours of recordings (4,172,012 time points) stored in multiple CSV files. These recordings were divided into non-overlapping 1.5-hour segments (1,080 time points per segment), excluding incomplete segments. This process yielded **3,817 segments from 64 patients**, representing 5,725.50 hours of recordings.

The segmented dataset contains naturally occurring missing values, with missing-data proportions of **5.44% for MAP** and **19.70% for ScO₂**. Most missing-data gaps were shorter than 1 minute, accounting for 66.88% and 85.21% of the total gaps in MAP and ScO₂, respectively.

**Dataset Summary:**

| Parameter | Value |
|-----------|-------|
| Number of patients | 64 |
| Sampling frequency | 0.2 Hz |
| Sampling interval | 5 seconds |
| Original recording duration | 5,794.46 hours |
| Total time points | 4,172,012 |
| Segment duration | 1.5 hours |
| Time points per segment | 1,080 |
| Total segments | 3,817 |
| Total segmented duration | 5,725.50 hours |
| Missing-data proportion (MAP) | 5.44% |
| Missing-data proportion (ScO₂) | 19.70% |

<img width="846" height="333" alt="image" src="https://github.com/user-attachments/assets/b21693db-8646-4b40-8193-f8779bc25270" />
Figure 1. The number of length-dependent (in min) data gaps of (a) ScO₂ and (b) MAP, as labeled individually by the number of missing gaps/incidents and corresponding per-centage for each time length. Note that the notation of [a, b) min denotes a time interval that includes values equal to or greater than “a” min but less than “b” min.

### Problem Definition

Given a multivariate time series $X \in \mathbb{R}^{T \times D}$, where $T$ represents the number of time points and $D$ represents the number of channels ($D = 2$ for MAP and ScO₂), the objective of imputation is to reconstruct missing values using conventional or deep learning-based methods.

Let $X$ denote the time-series data containing missing values and $X_{\mathrm{gt}}$ denote the corresponding ground-truth data. The imputation function $f(X)$ aims to minimize the reconstruction error, expressed as:

$$
\min_f \mathcal{L} = \left\|f(X) - X_{\mathrm{gt}}\right\|_1
$$

where $\|\cdot\|_1$ represents the L1 norm (sum of absolute errors).

Since ground-truth values are unavailable for naturally occurring missing data, artificial gaps are introduced into observed portions of the recordings. Reconstruction errors are then calculated only at these artificially masked positions to evaluate imputation performance.

### Selection of Imputation Methods

Nine imputation methods were selected for evaluation, including three conventional methods and six deep learning-based methods.

The three conventional methods are linear interpolation (LI), mean carry-forward (MCF), and last observation carried forward (LOCF).

The six deep learning-based methods are:

1. **NAOMI** – Non-Autoregressive Multiresolution Sequence Imputation ([Liu et al., 2019](https://arxiv.org/abs/1901.10946)).
2. **MTSIT** – Multivariate Time Series Imputation with Transformers ([Yıldız et al., 2022](https://doi.org/10.1109/LSP.2022.3224880)).
3. **CSDI** – Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation ([Tashiro et al., 2021](https://arxiv.org/abs/2107.03502)).
4. **TransMIT** – Transformer-Based Missing Data Imputation in Time Series ([Li et al., 2026](https://doi.org/10.1109/TIE.2026.3672736)).
5. **GP-VAE** – Deep Probabilistic Time Series Imputation ([Fortuin et al., 2020](https://proceedings.mlr.press/v108/fortuin20a.html)).
6. **SAITS** – Self-Attention-based Imputation for Time Series ([Du et al., 2023](https://doi.org/10.1016/j.eswa.2023.119619)).

These methods represent different modeling approaches, including recurrent neural networks, Transformer-based architectures, diffusion models, and variational autoencoders. Their implementations were developed based on the corresponding original publications.

### Performance Evaluation and Cross-Validation

We employed **5-fold patient-level stratified cross-validation** to evaluate the performance of different imputation methods. In each fold, patients were divided into training (72%), validation (8%), and test (20%) sets. All segments belonging to the same patient were assigned to the same set to prevent data leakage. Among the 64 patients, four test folds contained 13 patients each, while the remaining fold contained 12 patients. The validation set contained approximately 5 patients per fold.

After cross-validation, predictions from all five held-out test folds were aggregated to evaluate the overall imputation performance.

Two evaluation metrics were used: **mean absolute error (MAE)** and **mean absolute percentage error (MAPE)**, defined as follows:

**Mean Absolute Error (MAE):**

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**Mean Absolute Percentage Error (MAPE):**

$$
\mathrm{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i-\hat{y}_i|}{|y_i|+10^{-8}}
$$

where $y_i$ and $\hat{y}_i$ represent the ground-truth and imputed values at time point $i$, respectively, and $n$ is the total number of evaluated time points within the artificially generated gaps. The constant $10^{-8}$ is added to the denominator of MAPE to prevent division by zero.

Both metrics were calculated using the aggregated predictions from all five test folds, considering only artificially masked time points with available ground-truth values.
