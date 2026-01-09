"""
Wiener Filter implementation for noise reduction.
Complexity Level: High

The Wiener filter is an optimal linear filter that minimizes the mean square error
between the estimated and the desired signal. It adapts to the local signal and
noise characteristics, providing superior performance compared to simpler methods.
"""

import numpy as np
from scipy.fft import fft, ifft


def estimate_noise_psd(signal, sample_rate, noise_frames=10, frame_size=512, hop_size=256):
    """
    Estimate the noise Power Spectral Density (PSD) from the initial frames.

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        noise_frames: Number of frames to use for estimation
        frame_size: FFT frame size
        hop_size: Hop size between frames

    Returns:
        numpy array: Estimated noise PSD
    """
    noise_samples = signal[:frame_size * noise_frames]
    window = np.hanning(frame_size)
    noise_psds = []

    for i in range(noise_frames):
        start = i * hop_size
        end = start + frame_size
        if end > len(noise_samples):
            break

        frame = noise_samples[start:end]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        windowed_frame = frame * window
        spectrum = fft(windowed_frame)
        psd = np.abs(spectrum) ** 2
        noise_psds.append(psd)

    if noise_psds:
        noise_psd = np.mean(noise_psds, axis=0)
    else:
        noise_psd = np.ones(frame_size)

    return noise_psd


def compute_wiener_gain(signal_psd, noise_psd, epsilon=1e-10):
    """
    Compute the Wiener filter gain function.

    H(f) = S_ss(f) / (S_ss(f) + S_nn(f))

    where S_ss is the signal PSD and S_nn is the noise PSD.

    Args:
        signal_psd: Signal Power Spectral Density
        noise_psd: Noise Power Spectral Density
        epsilon: Small value to prevent division by zero

    Returns:
        numpy array: Wiener filter gain
    """
    # Estimate signal PSD (assuming signal PSD = observed PSD - noise PSD)
    clean_signal_psd = np.maximum(signal_psd - noise_psd, epsilon)

    # Compute Wiener gain
    wiener_gain = clean_signal_psd / (clean_signal_psd + noise_psd + epsilon)

    return wiener_gain


def apply_wiener_filter(signal, sample_rate, noise_psd=None, frame_size=512,
                        hop_size=256, smoothing=0.9):
    """
    Apply Wiener filtering for noise reduction.

    The Wiener filter is an optimal linear filter in the MSE sense. It computes
    a frequency-dependent gain based on the estimated signal and noise PSDs:

    H(f) = S_ss(f) / (S_ss(f) + S_nn(f))

    This gain is applied in the frequency domain to suppress noise while
    preserving the signal.

    Args:
        signal: Input noisy signal (numpy array)
        sample_rate: Sample rate in Hz
        noise_psd: Pre-computed noise PSD (if None, will be estimated)
        frame_size: FFT frame size (default: 512)
        hop_size: Hop size between frames (default: 256)
        smoothing: Temporal smoothing factor for gain (default: 0.9)

    Returns:
        numpy array: Denoised signal
    """
    # Estimate noise PSD if not provided
    if noise_psd is None:
        noise_psd = estimate_noise_psd(signal, sample_rate,
                                        frame_size=frame_size,
                                        hop_size=hop_size)

    num_samples = len(signal)
    output_signal = np.zeros(num_samples)
    window = np.hanning(frame_size)
    normalization = np.zeros(num_samples)

    # Previous gain for temporal smoothing
    prev_gain = np.ones(frame_size)

    num_frames = (num_samples - frame_size) // hop_size + 1

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        if end > num_samples:
            break

        # Extract and window the frame
        frame = signal[start:end] * window

        # Compute FFT
        spectrum = fft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # Compute signal PSD for this frame
        signal_psd = magnitude ** 2

        # Compute Wiener gain
        wiener_gain = compute_wiener_gain(signal_psd, noise_psd)

        # Apply temporal smoothing to reduce musical noise
        smoothed_gain = smoothing * prev_gain + (1 - smoothing) * wiener_gain
        prev_gain = smoothed_gain

        # Apply gain to magnitude spectrum
        filtered_magnitude = magnitude * smoothed_gain

        # Reconstruct complex spectrum
        reconstructed_spectrum = filtered_magnitude * np.exp(1j * phase)

        # Inverse FFT
        reconstructed_frame = np.real(ifft(reconstructed_spectrum))

        # Overlap-add
        output_signal[start:end] += reconstructed_frame * window
        normalization[start:end] += window ** 2

    # Normalize
    normalization[normalization < 1e-8] = 1e-8
    output_signal = output_signal / normalization

    return output_signal


def apply_wiener_filter_iterative(signal, sample_rate, noise_psd=None,
                                   frame_size=512, hop_size=256, iterations=2):
    """
    Apply iterative Wiener filtering for improved noise reduction.

    Multiple iterations can improve the estimate of the clean signal PSD,
    leading to better noise reduction. Each iteration uses the output of
    the previous iteration to refine the signal PSD estimate.

    Args:
        signal: Input noisy signal
        sample_rate: Sample rate in Hz
        noise_psd: Pre-computed noise PSD
        frame_size: FFT frame size
        hop_size: Hop size between frames
        iterations: Number of iterations (default: 2)

    Returns:
        numpy array: Denoised signal
    """
    # Estimate noise PSD if not provided
    if noise_psd is None:
        noise_psd = estimate_noise_psd(signal, sample_rate,
                                        frame_size=frame_size,
                                        hop_size=hop_size)

    current_signal = signal.copy()

    for iteration in range(iterations):
        # Update the signal PSD estimate based on current cleaned signal
        current_signal = apply_wiener_filter(current_signal, sample_rate,
                                              noise_psd, frame_size, hop_size)

    return current_signal


def parametric_wiener_filter(signal, sample_rate, noise_psd=None,
                              frame_size=512, hop_size=256,
                              alpha=0.98, beta=0.7, mu=0.98):
    """
    Parametric Wiener filter with decision-directed approach.

    Uses the decision-directed approach for a priori SNR estimation:
    - alpha: Smoothing factor for a priori SNR
    - beta: Gain function exponent
    - mu: Noise tracking parameter

    Args:
        signal: Input noisy signal
        sample_rate: Sample rate in Hz
        noise_psd: Pre-computed noise PSD
        frame_size: FFT frame size
        hop_size: Hop size between frames
        alpha: A priori SNR smoothing factor
        beta: Gain function exponent
        mu: Noise tracking parameter

    Returns:
        numpy array: Denoised signal
    """
    if noise_psd is None:
        noise_psd = estimate_noise_psd(signal, sample_rate,
                                        frame_size=frame_size,
                                        hop_size=hop_size)

    num_samples = len(signal)
    output_signal = np.zeros(num_samples)
    window = np.hanning(frame_size)
    normalization = np.zeros(num_samples)

    prev_clean_spectrum = np.zeros(frame_size)
    epsilon = 1e-10

    num_frames = (num_samples - frame_size) // hop_size + 1

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        if end > num_samples:
            break

        frame = signal[start:end] * window
        spectrum = fft(frame)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # A posteriori SNR
        gamma = (magnitude ** 2) / (noise_psd + epsilon)

        # A priori SNR using decision-directed approach
        if i == 0:
            xi = np.maximum(gamma - 1, 0)
        else:
            xi = alpha * (np.abs(prev_clean_spectrum) ** 2) / (noise_psd + epsilon) + \
                 (1 - alpha) * np.maximum(gamma - 1, 0)

        # Parametric Wiener gain
        gain = (xi / (xi + 1)) ** beta

        # Apply gain
        filtered_magnitude = magnitude * gain
        prev_clean_spectrum = filtered_magnitude * np.exp(1j * phase)

        reconstructed_frame = np.real(ifft(prev_clean_spectrum))

        output_signal[start:end] += reconstructed_frame * window
        normalization[start:end] += window ** 2

    normalization[normalization < 1e-8] = 1e-8
    output_signal = output_signal / normalization

    return output_signal

