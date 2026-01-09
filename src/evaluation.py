"""
Evaluation metrics module for Audio Noise Reduction Study.
Implements SNR, PSNR, and MSE calculations for comparing algorithm performance.
"""

import numpy as np


def calculate_snr(signal, noise):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    Args:
        signal: Clean/reference signal
        noise: Noise component (can be estimated as clean - denoised)

    Returns:
        float: SNR in decibels
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power < 1e-10:
        return float('inf')

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_snr_from_signals(clean_signal, noisy_signal):
    """
    Calculate SNR given clean and noisy signals.

    Args:
        clean_signal: Original clean signal
        noisy_signal: Noisy/degraded signal

    Returns:
        float: SNR in decibels
    """
    min_len = min(len(clean_signal), len(noisy_signal))
    clean = clean_signal[:min_len]
    noisy = noisy_signal[:min_len]

    noise_estimate = noisy - clean

    return calculate_snr(clean, noise_estimate)


def calculate_snr_improvement(clean_signal, noisy_signal, denoised_signal):
    """
    Calculate the SNR improvement after noise reduction.

    SNR_improvement = SNR_output - SNR_input

    Args:
        clean_signal: Original clean signal
        noisy_signal: Noisy input signal
        denoised_signal: Signal after noise reduction

    Returns:
        float: SNR improvement in decibels
    """
    min_len = min(len(clean_signal), len(noisy_signal), len(denoised_signal))
    clean = clean_signal[:min_len]
    noisy = noisy_signal[:min_len]
    denoised = denoised_signal[:min_len]

    input_noise = noisy - clean
    input_snr = calculate_snr(clean, input_noise)

    output_noise = denoised - clean
    output_snr = calculate_snr(clean, output_noise)

    snr_improvement = output_snr - input_snr

    return snr_improvement


def calculate_output_snr(clean_signal, denoised_signal):
    """
    Calculate the output SNR after noise reduction.

    Args:
        clean_signal: Original clean signal
        denoised_signal: Signal after noise reduction

    Returns:
        float: Output SNR in decibels
    """
    min_len = min(len(clean_signal), len(denoised_signal))
    clean = clean_signal[:min_len]
    denoised = denoised_signal[:min_len]

    residual_noise = denoised - clean
    output_snr = calculate_snr(clean, residual_noise)

    return output_snr


def calculate_mse(reference_signal, processed_signal):
    """
    Calculate the Mean Squared Error (MSE) between two signals.

    Args:
        reference_signal: Reference/clean signal
        processed_signal: Processed/denoised signal

    Returns:
        float: Mean Squared Error
    """
    min_len = min(len(reference_signal), len(processed_signal))
    reference = reference_signal[:min_len]
    processed = processed_signal[:min_len]

    mse = np.mean((reference - processed) ** 2)
    return mse


def calculate_psnr(reference_signal, processed_signal, max_value=1.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) in decibels.

    Args:
        reference_signal: Reference/clean signal
        processed_signal: Processed/denoised signal
        max_value: Maximum possible signal value (default: 1.0 for normalized audio)

    Returns:
        float: PSNR in decibels
    """
    mse = calculate_mse(reference_signal, processed_signal)

    if mse < 1e-10:
        return float('inf')

    psnr_db = 10 * np.log10((max_value ** 2) / mse)
    return psnr_db


def calculate_rmse(reference_signal, processed_signal):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Args:
        reference_signal: Reference/clean signal
        processed_signal: Processed/denoised signal

    Returns:
        float: Root Mean Squared Error
    """
    mse = calculate_mse(reference_signal, processed_signal)
    return np.sqrt(mse)


def evaluate_algorithm(clean_signal, noisy_signal, denoised_signal):
    """
    Compute all evaluation metrics for a single algorithm application.

    Args:
        clean_signal: Original clean signal
        noisy_signal: Noisy input signal
        denoised_signal: Signal after noise reduction

    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {
        'snr_improvement_db': calculate_snr_improvement(clean_signal, noisy_signal, denoised_signal),
        'output_snr_db': calculate_output_snr(clean_signal, denoised_signal),
        'mse': calculate_mse(clean_signal, denoised_signal),
        'psnr_db': calculate_psnr(clean_signal, denoised_signal),
        'rmse': calculate_rmse(clean_signal, denoised_signal),
        'noisy_mse': calculate_mse(clean_signal, noisy_signal),
        'noisy_psnr_db': calculate_psnr(clean_signal, noisy_signal)
    }

    return metrics


def evaluate_all_algorithms(clean_signal, noisy_signal, denoised_signals):
    """
    Evaluate multiple algorithms and return comparative metrics.

    Args:
        clean_signal: Original clean signal
        noisy_signal: Noisy input signal
        denoised_signals: Dictionary mapping algorithm names to denoised signals

    Returns:
        dict: Dictionary mapping algorithm names to their metrics
    """
    results = {}

    for algo_name, denoised_signal in denoised_signals.items():
        results[algo_name] = evaluate_algorithm(clean_signal, noisy_signal, denoised_signal)

    return results


def calculate_segmental_snr(clean_signal, processed_signal, frame_size=256, hop_size=128):
    """
    Calculate Segmental SNR (average SNR over short segments).

    Args:
        clean_signal: Original clean signal
        processed_signal: Processed signal
        frame_size: Frame size for segmentation
        hop_size: Hop size between frames

    Returns:
        float: Segmental SNR in decibels
    """
    min_len = min(len(clean_signal), len(processed_signal))
    clean = clean_signal[:min_len]
    processed = processed_signal[:min_len]

    noise = processed - clean

    segmental_snrs = []
    num_frames = (min_len - frame_size) // hop_size + 1

    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size

        clean_frame = clean[start:end]
        noise_frame = noise[start:end]

        frame_snr = calculate_snr(clean_frame, noise_frame)

        # Clip extreme values
        frame_snr = np.clip(frame_snr, -10, 35)
        segmental_snrs.append(frame_snr)

    if segmental_snrs:
        return np.mean(segmental_snrs)
    else:
        return 0.0

