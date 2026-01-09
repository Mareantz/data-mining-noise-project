"""
Dataset construction module for Audio Noise Reduction Study.
Handles loading audio files and creating synthetic noisy datasets.
"""

import numpy as np
import os
from scipy.io import wavfile
import warnings

# Suppress wavfile warnings about metadata
warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)


def load_audio(filepath):
    """
    Load an audio file and return the signal and sample rate.

    Args:
        filepath: Path to the audio file (.wav)

    Returns:
        tuple: (signal as numpy array normalized to [-1, 1], sample_rate)
    """
    try:
        sample_rate, signal = wavfile.read(filepath)

        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        elif signal.dtype == np.uint8:
            signal = (signal.astype(np.float32) - 128) / 128.0
        else:
            signal = signal.astype(np.float32)

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

        return signal, sample_rate
    except Exception as e:
        raise IOError(f"Error loading audio file {filepath}: {e}")


def save_audio(filepath, signal, sample_rate):
    """
    Save a signal as a WAV file.

    Args:
        filepath: Output path for the WAV file
        signal: Audio signal (numpy array, normalized to [-1, 1])
        sample_rate: Sample rate in Hz
    """
    signal = np.clip(signal, -1.0, 1.0)
    signal_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(filepath, sample_rate, signal_int16)


def calculate_signal_power(signal):
    return np.mean(signal ** 2)


def mix_audio_at_snr(clean_signal, noise_signal, target_snr_db):
    """
    Mix clean speech with noise at a specified Signal-to-Noise Ratio.

    Args:
        clean_signal: Clean speech signal (numpy array)
        noise_signal: Background noise signal (numpy array)
        target_snr_db: Target SNR in decibels

    Returns:
        tuple: (mixed_signal, scaled_noise)
    """
    min_len = min(len(clean_signal), len(noise_signal))
    clean = clean_signal[:min_len]
    noise = noise_signal[:min_len]

    clean_power = calculate_signal_power(clean)

    target_noise_power = clean_power / (10 ** (target_snr_db / 10))

    current_noise_power = calculate_signal_power(noise)
    if current_noise_power > 0:
        scaling_factor = np.sqrt(target_noise_power / current_noise_power)
    else:
        scaling_factor = 0

    scaled_noise = noise * scaling_factor
    mixed_signal = clean + scaled_noise

    max_val = np.max(np.abs(mixed_signal))
    if max_val > 1.0:
        mixed_signal = mixed_signal / max_val

    return mixed_signal, scaled_noise


def generate_synthetic_audio(duration_seconds=3.0, sample_rate=16000, frequency=440.0):
    """
    Generate a synthetic clean signal (sine wave with harmonics to simulate speech).
    Used when real audio files are not available.

    Args:
        duration_seconds: Duration of the signal in seconds
        sample_rate: Sample rate in Hz
        frequency: Base frequency in Hz

    Returns:
        numpy array: Synthetic signal
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)

    signal = (0.5 * np.sin(2 * np.pi * frequency * t) +
              0.3 * np.sin(2 * np.pi * frequency * 2 * t) +
              0.15 * np.sin(2 * np.pi * frequency * 3 * t) +
              0.05 * np.sin(2 * np.pi * frequency * 4 * t))

    envelope = np.abs(np.sin(2 * np.pi * 2 * t)) ** 0.5
    signal = signal * envelope

    signal = signal / np.max(np.abs(signal)) * 0.9

    return signal


def generate_noise_signal(duration_seconds=3.0, sample_rate=16000, noise_type='white'):
    """
    Generate synthetic noise signal.

    Args:
        duration_seconds: Duration in seconds
        sample_rate: Sample rate in Hz
        noise_type: Type of noise ('white', 'pink', 'brown')

    Returns:
        numpy array: Noise signal
    """
    num_samples = int(sample_rate * duration_seconds)

    if noise_type == 'white':
        noise = np.random.randn(num_samples)
    elif noise_type == 'pink':
        noise = np.zeros(num_samples)
        num_rows = 16
        rows = np.zeros(num_rows)
        for i in range(num_samples):
            noise[i] = np.sum(rows)
            for j in range(num_rows):
                if (i + 1) % (2 ** j) == 0:
                    rows[j] = np.random.randn()
        noise = noise / num_rows
    elif noise_type == 'brown':
        noise = np.cumsum(np.random.randn(num_samples))
        noise = noise - np.mean(noise)
    else:
        noise = np.random.randn(num_samples)

    # Normalize
    noise = noise / np.max(np.abs(noise)) * 0.9

    return noise


def generate_dataset(clean_dir, noise_dir, output_dir, snr_levels, sample_rate=16000):
    """
    Generate a synthetic dataset by mixing clean speech with noise at various SNR levels.
    If no real audio files are found, generates synthetic signals.

    Args:
        clean_dir: Directory containing clean speech WAV files
        noise_dir: Directory containing noise WAV files
        output_dir: Directory to save mixed audio files
        snr_levels: List of target SNR levels in dB (e.g., [0, 5, 10, 15])
        sample_rate: Target sample rate

    Returns:
        list: List of dictionaries containing file info and paths
    """
    dataset = []

    clean_files = []
    if os.path.exists(clean_dir):
        clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]

    noise_files = []
    if os.path.exists(noise_dir):
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]

    if not clean_files or not noise_files:
        print("No audio files found. Generating synthetic dataset...")

        clean_signals = [
            ('synthetic_speech_1', generate_synthetic_audio(3.0, sample_rate, 200)),
            ('synthetic_speech_2', generate_synthetic_audio(3.0, sample_rate, 300)),
            ('synthetic_speech_3', generate_synthetic_audio(3.0, sample_rate, 250)),
        ]

        noise_signals = [
            ('white_noise', generate_noise_signal(3.0, sample_rate, 'white')),
            ('pink_noise', generate_noise_signal(3.0, sample_rate, 'pink')),
        ]

        for clean_name, clean_signal in clean_signals:
            clean_path = os.path.join(clean_dir, f"{clean_name}.wav")
            save_audio(clean_path, clean_signal, sample_rate)
            print(f"  Saved clean signal: {clean_path}")

        for noise_name, noise_signal in noise_signals:
            noise_path = os.path.join(noise_dir, f"{noise_name}.wav")
            save_audio(noise_path, noise_signal, sample_rate)
            print(f"  Saved noise signal: {noise_path}")

        for clean_name, clean_signal in clean_signals:
            for noise_name, noise_signal in noise_signals:
                for snr in snr_levels:
                    mixed_signal, scaled_noise = mix_audio_at_snr(clean_signal, noise_signal, snr)

                    filename = f"{clean_name}_{noise_name}_snr{snr}dB.wav"
                    output_path = os.path.join(output_dir, filename)

                    save_audio(output_path, mixed_signal, sample_rate)

                    dataset.append({
                        'filename': filename,
                        'clean_source': clean_name,
                        'noise_source': noise_name,
                        'snr_db': snr,
                        'mixed_path': output_path,
                        'clean_signal': clean_signal,
                        'noise_signal': scaled_noise,
                        'mixed_signal': mixed_signal,
                        'sample_rate': sample_rate
                    })

        print(f"Generated {len(dataset)} synthetic audio samples.")
    else:
        print(f"Found {len(clean_files)} clean files and {len(noise_files)} noise files.")

        for clean_file in clean_files:
            clean_path = os.path.join(clean_dir, clean_file)
            clean_signal, sr = load_audio(clean_path)

            for noise_file in noise_files:
                noise_path = os.path.join(noise_dir, noise_file)
                noise_signal, _ = load_audio(noise_path)

                for snr in snr_levels:
                    mixed_signal, scaled_noise = mix_audio_at_snr(clean_signal, noise_signal, snr)

                    clean_name = os.path.splitext(clean_file)[0]
                    noise_name = os.path.splitext(noise_file)[0]
                    filename = f"{clean_name}_{noise_name}_snr{snr}dB.wav"
                    output_path = os.path.join(output_dir, filename)

                    save_audio(output_path, mixed_signal, sr)

                    dataset.append({
                        'filename': filename,
                        'clean_source': clean_name,
                        'noise_source': noise_name,
                        'snr_db': snr,
                        'mixed_path': output_path,
                        'clean_signal': clean_signal[:len(mixed_signal)],
                        'noise_signal': scaled_noise,
                        'mixed_signal': mixed_signal,
                        'sample_rate': sr
                    })

        print(f"Generated {len(dataset)} mixed audio samples from real files.")

    return dataset

