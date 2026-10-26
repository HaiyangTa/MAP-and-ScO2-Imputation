# Traditional and Deep learning method for MAP and ScO2 imputation
This repository is created with the purpose to save and display the implementation of traditional and deep learning methods in MAP and SCO2 impotation. 
### Introduction
Mean arterial pressure (MAP) and regional cerebral oxygen saturation (ScO₂) are important physiological signals for assessing cerebral perfusion, oxygenation, and cerebrovascular autoregulation following pediatric cardiac arrest. However, clinical recordings frequently contain missing data due to sensor disconnections, patient motion, and equipment-related interruptions, potentially compromising downstream physiological analyses. Although conventional and deep learning-based imputation methods have been widely investigated, their performance on higher-frequency physiological recordings and their impact on downstream analyses remain insufficiently explored.

This study comprehensively evaluates conventional and deep learning-based imputation methods for reconstructing missing MAP and ScO₂ data sampled at 0.2 Hz. Specifically, we compare their reconstruction accuracy, investigate whether they achieve stringent MAE thresholds of 2 mmHg for MAP and 2 percentage points for ScO₂, and assess their ability to preserve the dynamic relationship between MAP and ScO₂ through wavelet transform coherence (WTC) analysis.

