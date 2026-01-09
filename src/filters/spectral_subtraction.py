"""
Spectral Subtraction noise reduction implementation.
Complexity Level: Medium

Spectral Subtraction is a statistical method that estimates the noise spectrum
and subtracts it from the noisy signal in the frequency domain. This method
assumes that noise is additive and stationary.
"""

import numpy as np
from scipy.fft import fft, ifft


def estimate_noise_spectrum(signal, sample_rate, noise_frames=10, frame_size=512, hop_size=256):
    """
    Estimate the noise spectrum from the first few frames of the signal.
    Assumes the beginning of the signal contains primarily noise (silence/pause).

    Args:
        signal: Input signal
        sample_rate: Sample rate in Hz
        noise_frames: Number of frames to use for noise estimation
        frame_size: FFT frame size
        hop_size: Hop size between frames

    Returns:
        numpy array: Estimated noise magnitude spectrum
    """
    noise_samples = signal[:frame_size * noise_frames]

    window = np.hanning(frame_size)
    noise_spectra = []

    for i in range(noise_frames):
        start = i * hop_size
        end = start + frame_size
        if end > len(noise_samples):
            break

        frame = noise_samples[start:end]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))

        windowed_frame = frame * window
        spectrum = np.abs(fft(windowed_frame))
        noise_spectra.append(spectrum)

    if noise_spectra:
        noise_spectrum = np.mean(noise_spectra, axis=0)
    else:
        noise_spectrum = np.zeros(frame_size)

    return noise_spectrum


def spectral_subtract(signal, sample_rate, noise_spectrum=None, frame_size=512,
                      hop_size=256, alpha=2.0, beta=0.01):
    """
    Apply Spectral Subtraction for noise reduction.

    Args:
        signal: Input noisy signal (numpy array)
        sample_rate: Sample rate in Hz
        noise_spectrum: Pre-computed noise spectrum (if None, will be estimated)
        frame_size: FFT frame size (default: 512)
        hop_size: Hop size between frames (default: 256)
        alpha: Over-subtraction factor (default: 2.0)
        beta: Spectral floor parameter to prevent musical noise (default: 0.01)

    Returns:
        numpy array: Denoised signal
    """
    if noise_spectrum is None:
        noise_spectrum = estimate_noise_spectrum(signal, sample_rate,
                                                  frame_size=frame_size,
                                                  hop_size=hop_size)

    num_samples = len(signal)
    output_signal = np.zeros(num_samples)
    window = np.hanning(frame_size)

    normalization = np.zeros(num_samples)

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

        subtracted_magnitude = magnitude - alpha * noise_spectrum

        spectral_floor = beta * magnitude
        subtracted_magnitude = np.maximum(subtracted_magnitude, spectral_floor)

        reconstructed_spectrum = subtracted_magnitude * np.exp(1j * phase)

        reconstructed_frame = np.real(ifft(reconstructed_spectrum))

        output_signal[start:end] += reconstructed_frame * window
        normalization[start:end] += window ** 2

    normalization[normalization < 1e-8] = 1e-8
    output_signal = output_signal / normalization

    return output_signal


def spectral_subtract_enhanced(signal, sample_rate, noise_spectrum=None, frame_size=512,
                                hop_size=256, alpha_range=(1.0, 5.0), beta=0.02):
    """
    Enhanced Spectral Subtraction with adaptive over-subtraction.

    Args:
        signal: Input noisy signal
        sample_rate: Sample rate in Hz
        noise_spectrum: Pre-computed noise spectrum
        frame_size: FFT frame size
        hop_size: Hop size between frames
        alpha_range: Range for adaptive alpha (min, max)
        beta: Spectral floor parameter

    Returns:
        numpy array: Denoised signal
    """
    if noise_spectrum is None:
        noise_spectrum = estimate_noise_spectrum(signal, sample_rate,
                                                  frame_size=frame_size,
                                                  hop_size=hop_size)

    num_samples = len(signal)
    output_signal = np.zeros(num_samples)
    window = np.hanning(frame_size)
    normalization = np.zeros(num_samples)

    alpha_min, alpha_max = alpha_range
    noise_power = np.mean(noise_spectrum ** 2)

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

        frame_power = np.mean(magnitude ** 2)
        local_snr = 10 * np.log10(frame_power / (noise_power + 1e-10))

        alpha = alpha_max - (alpha_max - alpha_min) * np.clip(local_snr / 20, 0, 1)

        subtracted_magnitude = magnitude - alpha * noise_spectrum
        spectral_floor = beta * magnitude
        subtracted_magnitude = np.maximum(subtracted_magnitude, spectral_floor)

        reconstructed_spectrum = subtracted_magnitude * np.exp(1j * phase)
        reconstructed_frame = np.real(ifft(reconstructed_spectrum))

        output_signal[start:end] += reconstructed_frame * window
        normalization[start:end] += window ** 2

    normalization[normalization < 1e-8] = 1e-8
    output_signal = output_signal / normalization

    return output_signal

