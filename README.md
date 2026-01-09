# Comparative Data Mining Study of Audio Noise Reduction Algorithms

## Project Overview
This project implements a complete data mining pipeline to evaluate and compare the efficacy of different noise-cancellation algorithms on audio data. The goal is to quantitatively determine which algorithmic approach provides the best signal restoration when handling noisy speech data.

## Project Structure
The workspace is organized as follows:
- `main.py`: The entry point script that orchestrates the entire pipeline.
- `src/`: Source code modules.
  - `dataset.py`: Handles loading audio, generating synthetic noise, and mixing signals.
  - `filters/`: Contains the implementations of the noise reduction algorithms.
  - `evaluation.py`: Computes performance metrics (SNR, MSE, PSNR).
  - `visualization.py`: Generates charts and summary reports.
- `data/`: Directory for processing audio files (Ignored in Git).
  - `clean/`: Original clean speech files (ground truth).
  - `noise/`: Background noise files.
  - `mixed/`: The input dataset (Clean + Noise at specific SNRs).
  - `denoised/`: The output from the algorithms.
- `output/`: Results (Ignored in Git).
  - `charts/`: Visualizations of the performance metrics.
  - `results.csv`: Detailed metrics for every file processed.
  - `summary.txt`: A text summary of the findings.
- `MS-SNSD/`: (External Dependency) The Microsoft Scalable Noisy Speech Dataset repository. (Ignored in Git due to size).

## Algorithms Explanation and Efficiency

We compare three distinct tiers of noise reduction algorithms:

### 1. Low Complexity: Butterworth Low-Pass Filter
- **What it does**: A frequency-domain filter that allows signals below a certain cutoff frequency (e.g., 4000 Hz) to pass while attenuating frequencies above it. It assumes that human speech lies mostly in lower frequencies and noise dominates the higher frequencies.
- **Efficiency**: **Very High**. It uses a recursive (IIR) structure which requires very few calculations per sample. Complexity is O(N).
- **Cons**: It is a "blunt instrument". It removes high-frequency noise well but also muffles speech (removing "s" and "f" sounds) and fails completely if noise is in the same frequency range as speech.

### 2. Medium Complexity: Spectral Subtraction
- **What it does**: A statistical method that operates in the frequency domain. It takes the Short-Time Fourier Transform (STFT) of the noisy signal, estimates the noise spectrum (usually by averaging the spectrum during silent periods), and subtracts this estimated noise from the noisy speech magnitude spectrum.
- **Efficiency**: **Medium**. Requires performing FFT and Inverse FFT operations. Complexity is O(N log N).
- **Cons**: Can introduce "musical noise" (metallic chirping artifacts) caused by random peaks in the noise spectrum that remain after subtraction.

### 3. High Complexity: Wiener Filter
- **What it does**: An adaptive optimal filter. It attempts to minimize the Mean Squared Error between the estimated signal and the true clean signal. Unlike spectral subtraction which just subtracts a fixed amount, the Wiener filter scales each frequency component based on the estimated Signal-to-Noise Ratio (SNR) at that specific frequency. If a frequency bin has high SNR, it is kept; if it has low SNR, it is attenuated.
- **Efficiency**: **High (computationally intensive)**. Involves block processing, FFTs, and continuous statistical estimation of signal and noise power spectra. Complexity is O(N log N).
- **Cons**: Computationally expensive. Performance depends heavily on the accuracy of the noise estimation.

## Metrics & Analysis

To evaluate performance, we calculate the following metrics for every audio file:

### 1. SNR (Signal-to-Noise Ratio)
- **Definition**: A measure of signal strength relative to background noise.
- **Calculation**: 
  $$ SNR_{dB} = 10 \cdot \log_{10}\left(\frac{\sum (\text{Clean Signal})^2}{\sum (\text{Clean Signal} - \text{Processed Signal})^2}\right) $$
- **Interpretation**: Higher is better. A positive **SNR Improvement** means the algorithm successfully reduced noise without destroying the speech.

### 2. MSE (Mean Squared Error)
- **Definition**: The average squared difference between the original clean signal and the denoised signal.
- **Calculation**: 
  $$ MSE = \frac{1}{N} \sum_{i=1}^{N} (clean_i - denoised_i)^2 $$
- **Interpretation**: Lower is better. Measures the raw "distance" between the result and the ideal.

### 3. PSNR (Peak Signal-to-Noise Ratio)
- **Definition**: Ratio between the maximum possible power of a signal and the power of corrupting noise.
- **Calculation**: 
  $$ PSNR_{dB} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right) $$
  *(Where MAX is the maximum amplitude of the signal, usually 1.0 for normalized audio)*
- **Interpretation**: Higher is better. Commonly used to assess quality reconstruction.

## How to Run

### 1. Prerequisite: Download MS-SNSD
Because the audio dataset is very large (~6GB), it is excluded from this Git repository. You must download it manually to run the pipeline.

1.  Clone the repository: [https://github.com/microsoft/MS-SNSD](https://github.com/microsoft/MS-SNSD)
2.  Place the `MS-SNSD` folder in the root of this project.
3.  Ensure it contains the `clean_train/` and `noise_train/` subfolders.

### 2. Run the Pipeline
Once the dataset is in place, run the main script:

```bash
python main.py
```

### 3. Review Output
The script will:
-   Select random samples from `MS-SNSD` and copy them to `data/`.
-   Generate mixed noisy audio files in `data/mixed/`.
-   Apply all three algorithms to every file.
-   Save denoised audio to `data/denoised/`.
-   Generate charts and summary reports in `output/`.

