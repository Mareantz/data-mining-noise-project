"""
Low-pass Butterworth Filter implementation.
Complexity Level: Low

This filter removes high-frequency noise by allowing only frequencies below
a specified cutoff to pass through. Simple but effective for certain types
of noise with high-frequency characteristics.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def design_butterworth_lowpass(cutoff_freq, sample_rate, order=5):
    """
    Design a Butterworth low-pass filter.

    Args:
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sample rate of the signal in Hz
        order: Filter order (higher = sharper cutoff)

    Returns:
        tuple: (b, a) filter coefficients
    """
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Ensure normalized cutoff is within valid range
    normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)

    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_butterworth(signal, sample_rate, cutoff_freq=4000, order=5):
    """
    Apply a low-pass Butterworth filter to remove high-frequency noise.

    Args:
        signal: Input audio signal (numpy array)
        sample_rate: Sample rate in Hz
        cutoff_freq: Cutoff frequency in Hz (default: 4000 Hz)
        order: Filter order (default: 5)

    Returns:
        numpy array: Filtered signal
    """
    b, a = design_butterworth_lowpass(cutoff_freq, sample_rate, order)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def apply_bandpass_butterworth(signal, sample_rate, low_cutoff=300, high_cutoff=3400, order=5):
    """
    Apply a band-pass Butterworth filter.

    This filter keeps frequencies between low_cutoff and high_cutoff,
    which is typical for voice frequencies (300-3400 Hz for telephone quality).

    Args:
        signal: Input audio signal (numpy array)
        sample_rate: Sample rate in Hz
        low_cutoff: Low cutoff frequency in Hz
        high_cutoff: High cutoff frequency in Hz
        order: Filter order

    Returns:
        numpy array: Filtered signal
    """
    nyquist_freq = 0.5 * sample_rate
    low = low_cutoff / nyquist_freq
    high = high_cutoff / nyquist_freq

    # Ensure normalized frequencies are within valid range
    low = np.clip(low, 0.01, 0.99)
    high = np.clip(high, 0.01, 0.99)

    if low >= high:
        low = high - 0.1

    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

