# GMM-HMM

1. Based on the experimental manual, submit the experiment report and use the features extracted in Assignment 1 as input. Observe the recognition results. If the features you extracted differ significantly from those returned by the platform interface, analyze the possible reasons.

2. Using Maximum Likelihood Estimation method, estmate the parameters of mean μ in a multivariate Gaussian model given a set of sampled data $X={x_1, x_2, ..., x_n}$. The pdf of the multivariate Gaussian model is:

$$
p(\mathbf{x} | \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$

```
GMM_HMM/
├── assets/                  # Images used in Report.md
├── datas.zip                # Dataset
├── Features.npy             # Extracted MFCC
├── GMM_HMM.ipynb            # Model training and testing code
├── hmm_gmm_model.pkl        # HMM-GMM model file
├── MFCC_Extraction.ipynb    # MFCC feature extraction code
├── MLE.pdf                  # Maximum Likelihood Estimation for the Mean μ in a Multivariate Gaussian Model
├── README.md                # Readme documentation
├── Record.wav               # A sample audio recording
└── Report.md                # Experiment report
```

**The complete code and execution results can be found in `GMM_HMM.ipynb`.**