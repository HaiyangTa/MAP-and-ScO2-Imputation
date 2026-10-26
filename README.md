# Conventional and Deep Learning Methods for MAP and ScO₂ Imputation

This repository provides implementations of conventional and deep learning-based methods for imputing missing data in mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) recordings.

### Introduction

Mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) are important physiological signals for assessing cerebral perfusion, oxygenation, and cerebrovascular autoregulation following pediatric cardiac arrest. However, clinical recordings frequently contain missing data due to sensor disconnections, patient motion, and equipment-related interruptions, potentially compromising downstream physiological analyses. Although conventional and deep learning-based imputation methods have been widely investigated, their performance on higher-frequency physiological recordings and their impact on downstream analyses remain insufficiently explored.

This study systematically evaluates conventional and deep learning-based imputation methods for reconstructing missing MAP and ScO₂ data sampled at 0.2 Hz. Specifically, we compare their reconstruction accuracy, investigate whether they achieve stringent mean absolute error (MAE) thresholds of less than 2 mmHg for MAP and 2 percentage points for ScO₂, and assess their ability to preserve the dynamic physiological relationship between MAP and ScO₂ through wavelet transform coherence (WTC) analysis.

