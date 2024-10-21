# MFCC Extraction

Extract acoustic features (MFCC) for a segment of speech and compare the results with the output of MFCC function provided by Python package.

Processing steps include:

* Import Packages and Load Audio
* Pre-emphasis
* Windowing
* Short-Time Fourier Transform (STFT)
* Mel-filter Bank
* Log Transformation
* Discrete Cosine Transform (DCT)
* Dynamic Feature Extraction
* Feature Transformation
* Principal Component Analysis (PCA)

```
MFCC_Extraction/
├── assets/                  # Images used in Report.md
├── MFCC_Extraction.ipynb    # Code and execution results
├── README.md                # Readme documentation
├── Record.wav               # A sample audio recording
└── Report.md                # Report
```

**The complete code and execution results can be found in `MFCC_Extraction.ipynb`.**