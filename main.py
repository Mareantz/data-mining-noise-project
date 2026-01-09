"""
Comparative Data Mining Study of Audio Noise Reduction Algorithms

This script implements a complete data mining pipeline to evaluate and compare
the efficacy of multiple noise-cancellation algorithms on audio data.

Algorithms implemented:
1. Low Complexity: Low-pass Butterworth Filter
2. Medium Complexity: Spectral Subtraction
3. High Complexity: Wiener Filter

Metrics:
- SNR (Signal-to-Noise Ratio) Improvement
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)

Author: Data Mining Project
Date: January 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.dataset import generate_dataset, load_audio, save_audio
from src.filters.butterworth import apply_lowpass_butterworth
from src.filters.spectral_subtraction import spectral_subtract
from src.filters.wiener_filter import apply_wiener_filter
from src.evaluation import evaluate_algorithm
from src.visualization import (
    plot_comparison_bar_chart,
    plot_multi_metric_comparison,
    plot_snr_vs_improvement,
    generate_summary_report,
    print_summary_to_console
)


# Configuration
CONFIG = {
    'clean_dir': 'data/clean',
    'noise_dir': 'data/noise',
    'mixed_dir': 'data/mixed',
    'denoised_dir': 'data/denoised',  # NEW: Directory for denoised outputs
    'output_dir': 'output',
    'charts_dir': 'output/charts',
    'snr_levels': [0, 5, 10, 15],  # SNR levels in dB
    'sample_rate': 16000,  # Sample rate for audio processing
    'butterworth_cutoff': 4000,  # Cutoff frequency for Butterworth filter
    'butterworth_order': 5,  # Filter order
}


def copy_from_local_repo():
    """Copy audio files from the local MS-SNSD folder."""
    repo_path = 'MS-SNSD'
    if not os.path.exists(repo_path):
        print(f"Error: Local {repo_path} folder not found.")
        sys.exit(1)

    clean_src = os.path.join(repo_path, 'clean_train')
    noise_src = os.path.join(repo_path, 'noise_train')

    clean_dest = CONFIG['clean_dir']
    noise_dest = CONFIG['noise_dir']

    # Clean files
    if os.path.exists(clean_src):
        files = sorted([f for f in os.listdir(clean_src) if f.endswith('.wav')])[:10]
        if files:
            print(f"  Copying {len(files)} clean files from local MS-SNSD...")
            for f in files:
                shutil.copy(os.path.join(clean_src, f), clean_dest)

    # Noise files
    if os.path.exists(noise_src):
        files = sorted([f for f in os.listdir(noise_src) if f.endswith('.wav')])[:5]
        if files:
            print(f"  Copying {len(files)} noise files from local MS-SNSD...")
            for f in files:
                shutil.copy(os.path.join(noise_src, f), noise_dest)


def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        CONFIG['clean_dir'],
        CONFIG['noise_dir'],
        CONFIG['mixed_dir'],
        CONFIG['denoised_dir'],
        CONFIG['output_dir'],
        CONFIG['charts_dir']
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def apply_noise_reduction_algorithms(sample):
    """
    Apply all three noise reduction algorithms to a single sample.

    Args:
        sample: Dictionary containing sample data

    Returns:
        dict: Dictionary mapping algorithm names to denoised signals
    """
    mixed_signal = sample['mixed_signal']
    sample_rate = sample['sample_rate']

    denoised_signals = {}

    # 1. Low Complexity: Butterworth Low-pass Filter
    try:
        denoised_butterworth = apply_lowpass_butterworth(
            mixed_signal,
            sample_rate,
            cutoff_freq=CONFIG['butterworth_cutoff'],
            order=CONFIG['butterworth_order']
        )
        denoised_signals['Butterworth Filter'] = denoised_butterworth
    except Exception as e:
        print(f"  Warning: Butterworth filter failed: {e}")
        denoised_signals['Butterworth Filter'] = mixed_signal.copy()

    # 2. Medium Complexity: Spectral Subtraction
    try:
        denoised_spectral = spectral_subtract(
            mixed_signal,
            sample_rate,
            frame_size=512,
            hop_size=256,
            alpha=2.0,
            beta=0.02
        )
        denoised_signals['Spectral Subtraction'] = denoised_spectral
    except Exception as e:
        print(f"  Warning: Spectral Subtraction failed: {e}")
        denoised_signals['Spectral Subtraction'] = mixed_signal.copy()

    # 3. High Complexity: Wiener Filter
    try:
        denoised_wiener = apply_wiener_filter(
            mixed_signal,
            sample_rate,
            frame_size=512,
            hop_size=256,
            smoothing=0.9
        )
        denoised_signals['Wiener Filter'] = denoised_wiener
    except Exception as e:
        print(f"  Warning: Wiener filter failed: {e}")
        denoised_signals['Wiener Filter'] = mixed_signal.copy()

    return denoised_signals


def process_dataset(dataset, save_denoised=True):
    """
    Process all samples in the dataset through all algorithms.

    Args:
        dataset: List of sample dictionaries
        save_denoised: Whether to save denoised audio files

    Returns:
        pd.DataFrame: Results DataFrame with all metrics
    """
    results = []
    denoised_dir = CONFIG['denoised_dir']

    total_samples = len(dataset)
    print(f"\nProcessing {total_samples} samples...")

    for idx, sample in enumerate(dataset):
        print(f"  [{idx+1}/{total_samples}] Processing: {sample['filename']}")

        # Apply all algorithms
        denoised_signals = apply_noise_reduction_algorithms(sample)

        # Evaluate each algorithm and optionally save denoised outputs
        for algo_name, denoised_signal in denoised_signals.items():
            metrics = evaluate_algorithm(
                sample['clean_signal'],
                sample['mixed_signal'],
                denoised_signal
            )

            # Create safe algorithm name for filename
            algo_safe = algo_name.replace(' ', '_').lower()
            base_name = os.path.splitext(sample['filename'])[0]
            denoised_filename = f"{base_name}_{algo_safe}.wav"
            denoised_path = os.path.join(denoised_dir, denoised_filename)

            # Save denoised audio file
            if save_denoised:
                save_audio(denoised_path, denoised_signal, sample['sample_rate'])

            result = {
                'filename': sample['filename'],
                'clean_source': sample['clean_source'],
                'noise_source': sample['noise_source'],
                'snr_level': sample['snr_db'],
                'algorithm': algo_name,
                'snr_improvement_db': metrics['snr_improvement_db'],
                'output_snr_db': metrics['output_snr_db'],
                'mse': metrics['mse'],
                'psnr_db': metrics['psnr_db'],
                'rmse': metrics['rmse'],
                'noisy_mse': metrics['noisy_mse'],
                'noisy_psnr_db': metrics['noisy_psnr_db'],
                'denoised_file': denoised_filename  # Track output file
            }
            results.append(result)

    if save_denoised:
        print(f"\n  Denoised audio files saved to: {denoised_dir}/")

    return pd.DataFrame(results)


def save_results_csv(results_df, output_path):
    """Save results DataFrame to CSV."""
    try:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    except PermissionError:
        # If file is locked, use timestamped filename
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        alt_path = output_path.replace('.csv', f'_{timestamp}.csv')
        results_df.to_csv(alt_path, index=False)
        print(f"\nResults saved to: {alt_path} (original file was locked)")


def generate_all_visualizations(results_df):
    """Generate all visualization charts."""
    charts_dir = CONFIG['charts_dir']

    print("\nGenerating visualizations...")

    # 1. SNR Improvement comparison bar chart
    plot_comparison_bar_chart(
        results_df,
        'snr_improvement_db',
        os.path.join(charts_dir, 'snr_improvement_comparison.png'),
        title='SNR Improvement by Algorithm and Input SNR Level'
    )

    # 2. PSNR comparison bar chart
    plot_comparison_bar_chart(
        results_df,
        'psnr_db',
        os.path.join(charts_dir, 'psnr_comparison.png'),
        title='PSNR by Algorithm and Input SNR Level'
    )

    # 3. MSE comparison bar chart
    plot_comparison_bar_chart(
        results_df,
        'mse',
        os.path.join(charts_dir, 'mse_comparison.png'),
        title='MSE by Algorithm and Input SNR Level'
    )

    # 4. Multi-metric comparison
    plot_multi_metric_comparison(
        results_df,
        ['snr_improvement_db', 'psnr_db', 'mse'],
        os.path.join(charts_dir, 'multi_metric_comparison.png')
    )

    # 5. SNR improvement trend
    plot_snr_vs_improvement(
        results_df,
        os.path.join(charts_dir, 'snr_improvement_trend.png')
    )


def main():
    """Main entry point for the noise reduction study."""
    print("=" * 70)
    print("COMPARATIVE DATA MINING STUDY OF AUDIO NOISE REDUCTION ALGORITHMS")
    print("=" * 70)

    # Setup directories
    setup_directories()

    # Step 0: Setup Data from Local MS-SNSD
    print("\n[STEP 0] Setting up Audio Data")
    print("-" * 50)

    clean_files = [f for f in os.listdir(CONFIG['clean_dir']) if f.endswith('.wav')] if os.path.exists(CONFIG['clean_dir']) else []

    if not clean_files:
        copy_from_local_repo()
        clean_files = [f for f in os.listdir(CONFIG['clean_dir']) if f.endswith('.wav')]

    print(f"✓ Using {len(clean_files)} clean audio files")

    # Step 1: Generate Dataset
    print("\n[STEP 1] Generating Mixed Dataset")
    print("-" * 50)
    dataset = generate_dataset(
        clean_dir=CONFIG['clean_dir'],
        noise_dir=CONFIG['noise_dir'],
        output_dir=CONFIG['mixed_dir'],
        snr_levels=CONFIG['snr_levels'],
        sample_rate=CONFIG['sample_rate']
    )

    if not dataset:
        print("Error: No dataset generated. Exiting.")
        return

    # Step 2: Process Dataset
    print("\n[STEP 2] Applying Noise Reduction Algorithms")
    print("-" * 50)
    print("Algorithms being evaluated:")
    print("  1. Butterworth Filter (Low Complexity)")
    print("  2. Spectral Subtraction (Medium Complexity)")
    print("  3. Wiener Filter (High Complexity)")

    results_df = process_dataset(dataset, save_denoised=True)

    # Step 3: Save Results
    print("\n[STEP 3] Saving Results")
    print("-" * 50)
    csv_path = os.path.join(CONFIG['output_dir'], 'results.csv')
    save_results_csv(results_df, csv_path)

    # Step 4: Generate Visualizations
    print("\n[STEP 4] Generating Visualizations")
    print("-" * 50)
    generate_all_visualizations(results_df)

    # Step 5: Generate Summary Report
    print("\n[STEP 5] Generating Summary Report")
    print("-" * 50)
    report_path = os.path.join(CONFIG['output_dir'], 'summary.txt')
    report_text = generate_summary_report(results_df, report_path)

    # Step 6: Print Console Summary
    print_summary_to_console(results_df)

    # Print the full report to console as well
    print("\n" + report_text)

    print("\n" + "=" * 70)
    print("STUDY COMPLETE")
    print("=" * 70)
    print(f"\nOutput files generated:")
    print(f"  - Results CSV: {csv_path}")
    print(f"  - Summary Report: {report_path}")
    print(f"  - Charts: {CONFIG['charts_dir']}/")
    print(f"  - Denoised Audio: {CONFIG['denoised_dir']}/")
    print(f"\nAudio files for comparison:")
    print(f"  - Clean originals: {CONFIG['clean_dir']}/")
    print(f"  - Noise samples: {CONFIG['noise_dir']}/")
    print(f"  - Noisy (mixed): {CONFIG['mixed_dir']}/")
    print(f"  - Denoised outputs: {CONFIG['denoised_dir']}/")
    print("\nThank you for using the Audio Noise Reduction Study tool!")


if __name__ == '__main__':
    main()

