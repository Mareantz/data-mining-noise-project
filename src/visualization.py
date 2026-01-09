"""
Visualization module for Audio Noise Reduction Study.
Creates charts and generates summary reports.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_comparison_bar_chart(results_df, metric, output_path, title=None):
    """
    Create a grouped bar chart comparing algorithm performance across SNR levels.

    Args:
        results_df: DataFrame with columns ['algorithm', 'snr_level', metric]
        metric: Column name for the metric to plot
        output_path: Path to save the chart image
        title: Chart title (optional)
    """
    algorithms = results_df['algorithm'].unique()
    snr_levels = sorted(results_df['snr_level'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(snr_levels))
    width = 0.25

    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']

    for i, algo in enumerate(algorithms):
        algo_data = results_df[results_df['algorithm'] == algo]
        values = []
        for snr in snr_levels:
            snr_data = algo_data[algo_data['snr_level'] == snr]
            if len(snr_data) > 0:
                values.append(snr_data[metric].mean())
            else:
                values.append(0)

        offset = (i - len(algorithms)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=algo, color=colors[i % len(colors)])

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Input SNR Level (dB)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison by Algorithm',
                    fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{snr} dB' for snr in snr_levels])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_multi_metric_comparison(results_df, metrics, output_path):
    """
    Create a multi-panel figure comparing multiple metrics.

    Args:
        results_df: DataFrame with results
        metrics: List of metric column names to plot
        output_path: Path to save the chart
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    algorithms = results_df['algorithm'].unique()
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        for algo in algorithms:
            algo_data = results_df[results_df['algorithm'] == algo][metric]
            means.append(algo_data.mean())
            stds.append(algo_data.std())

        bars = ax.bar(algorithms, means, color=colors[:len(algorithms)],
                     yerr=stds, capsize=5, alpha=0.8)

        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f'{mean:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    plt.suptitle('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved multi-metric chart: {output_path}")


def plot_snr_vs_improvement(results_df, output_path):
    """
    Create a line plot showing SNR improvement vs input SNR for each algorithm.

    Args:
        results_df: DataFrame with results
        output_path: Path to save the chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = results_df['algorithm'].unique()
    snr_levels = sorted(results_df['snr_level'].unique())

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    for i, algo in enumerate(algorithms):
        algo_data = results_df[results_df['algorithm'] == algo]
        means = []
        stds = []
        for snr in snr_levels:
            snr_data = algo_data[algo_data['snr_level'] == snr]['snr_improvement_db']
            means.append(snr_data.mean())
            stds.append(snr_data.std())

        ax.errorbar(snr_levels, means, yerr=stds, label=algo,
                   color=colors[i % len(colors)], marker=markers[i % len(markers)],
                   capsize=3, linewidth=2, markersize=8)

    ax.set_xlabel('Input SNR Level (dB)', fontsize=12)
    ax.set_ylabel('SNR Improvement (dB)', fontsize=12)
    ax.set_title('SNR Improvement vs Input SNR Level', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved SNR improvement chart: {output_path}")


def generate_summary_report(results_df, output_path):
    """
    Generate a text summary report interpreting the results.

    Args:
        results_df: DataFrame containing all evaluation results
        output_path: Path to save the summary text file
    """
    lines = []
    lines.append("=" * 70)
    lines.append("AUDIO NOISE REDUCTION ALGORITHMS - COMPARATIVE ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. DATASET SUMMARY")
    lines.append("-" * 50)
    total_samples = len(results_df)
    algorithms = results_df['algorithm'].unique()
    snr_levels = sorted(results_df['snr_level'].unique())
    lines.append(f"   Total processed samples: {total_samples}")
    lines.append(f"   Algorithms evaluated: {', '.join(algorithms)}")
    lines.append(f"   SNR levels tested: {snr_levels} dB")
    lines.append("")

    lines.append("2. OVERALL PERFORMANCE BY ALGORITHM")
    lines.append("-" * 50)

    algo_stats = []
    for algo in algorithms:
        algo_data = results_df[results_df['algorithm'] == algo]
        stats = {
            'algorithm': algo,
            'avg_snr_improvement': algo_data['snr_improvement_db'].mean(),
            'avg_psnr': algo_data['psnr_db'].mean(),
            'avg_mse': algo_data['mse'].mean()
        }
        algo_stats.append(stats)

        lines.append(f"\n   {algo}:")
        lines.append(f"     - Average SNR Improvement: {stats['avg_snr_improvement']:.2f} dB")
        lines.append(f"     - Average PSNR: {stats['avg_psnr']:.2f} dB")
        lines.append(f"     - Average MSE: {stats['avg_mse']:.6f}")

    lines.append("")

    lines.append("3. ALGORITHM RANKING")
    lines.append("-" * 50)

    algo_stats_df = pd.DataFrame(algo_stats)
    algo_stats_df = algo_stats_df.sort_values('avg_snr_improvement', ascending=False)

    lines.append("\n   Ranking by SNR Improvement (higher is better):")
    for rank, (_, row) in enumerate(algo_stats_df.iterrows(), 1):
        lines.append(f"     {rank}. {row['algorithm']}: {row['avg_snr_improvement']:.2f} dB")

    algo_stats_df = algo_stats_df.sort_values('avg_psnr', ascending=False)
    lines.append("\n   Ranking by PSNR (higher is better):")
    for rank, (_, row) in enumerate(algo_stats_df.iterrows(), 1):
        lines.append(f"     {rank}. {row['algorithm']}: {row['avg_psnr']:.2f} dB")

    algo_stats_df = algo_stats_df.sort_values('avg_mse', ascending=True)
    lines.append("\n   Ranking by MSE (lower is better):")
    for rank, (_, row) in enumerate(algo_stats_df.iterrows(), 1):
        lines.append(f"     {rank}. {row['algorithm']}: {row['avg_mse']:.6f}")

    lines.append("")

    lines.append("4. PERFORMANCE BY SNR LEVEL")
    lines.append("-" * 50)

    for snr in snr_levels:
        lines.append(f"\n   At {snr} dB input SNR:")
        snr_data = results_df[results_df['snr_level'] == snr]

        for algo in algorithms:
            algo_snr_data = snr_data[snr_data['algorithm'] == algo]
            avg_improvement = algo_snr_data['snr_improvement_db'].mean()
            avg_psnr = algo_snr_data['psnr_db'].mean()
            lines.append(f"     {algo}: SNR Improvement = {avg_improvement:.2f} dB, PSNR = {avg_psnr:.2f} dB")

    lines.append("")

    lines.append("5. CONCLUSION")
    lines.append("-" * 50)

    algo_stats_df = pd.DataFrame(algo_stats)
    best_by_snr = algo_stats_df.loc[algo_stats_df['avg_snr_improvement'].idxmax()]
    best_by_psnr = algo_stats_df.loc[algo_stats_df['avg_psnr'].idxmax()]
    best_by_mse = algo_stats_df.loc[algo_stats_df['avg_mse'].idxmin()]

    lines.append("")
    lines.append(f"   BEST ALGORITHM BY SNR IMPROVEMENT: {best_by_snr['algorithm']}")
    lines.append(f"   - Achieved an average improvement of {best_by_snr['avg_snr_improvement']:.2f} dB")

    algo_stats_sorted = algo_stats_df.sort_values('avg_snr_improvement', ascending=False)
    if len(algo_stats_sorted) > 1:
        margin = algo_stats_sorted.iloc[0]['avg_snr_improvement'] - algo_stats_sorted.iloc[1]['avg_snr_improvement']
        second_best = algo_stats_sorted.iloc[1]['algorithm']
        lines.append(f"   - Outperformed {second_best} by {margin:.2f} dB margin")

    lines.append("")
    lines.append(f"   BEST ALGORITHM BY PSNR: {best_by_psnr['algorithm']}")
    lines.append(f"   - Achieved an average PSNR of {best_by_psnr['avg_psnr']:.2f} dB")

    lines.append("")
    lines.append(f"   BEST ALGORITHM BY MSE: {best_by_mse['algorithm']}")
    lines.append(f"   - Achieved the lowest average MSE of {best_by_mse['avg_mse']:.6f}")

    lines.append("")
    lines.append("   OVERALL RECOMMENDATION:")

    votes = {}
    for algo in [best_by_snr['algorithm'], best_by_psnr['algorithm'], best_by_mse['algorithm']]:
        votes[algo] = votes.get(algo, 0) + 1
    overall_best = max(votes, key=votes.get)

    lines.append(f"   Based on the comprehensive evaluation across multiple metrics,")
    lines.append(f"   the {overall_best} algorithm demonstrates the best overall")
    lines.append(f"   performance for audio noise reduction in this study.")

    lines.append("")
    lines.append("   ALGORITHM-SPECIFIC INSIGHTS:")
    lines.append("   - Butterworth Filter (Low Complexity): Simple frequency-domain filtering,")
    lines.append("     effective for high-frequency noise but may remove speech content.")
    lines.append("   - Spectral Subtraction (Medium Complexity): Statistical approach that")
    lines.append("     preserves more speech characteristics but may introduce musical noise.")
    lines.append("   - Wiener Filter (High Complexity): Optimal MSE approach with adaptive")
    lines.append("     gain, providing the best balance between noise reduction and speech quality.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved summary report: {output_path}")

    return '\n'.join(lines)


def print_summary_to_console(results_df):
    """
    Print a formatted summary to the console.

    Args:
        results_df: DataFrame containing all evaluation results
    """
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 70)

    algorithms = results_df['algorithm'].unique()

    print("\nAverage Performance by Algorithm:")
    print("-" * 50)

    summary_data = []
    for algo in algorithms:
        algo_data = results_df[results_df['algorithm'] == algo]
        summary_data.append({
            'Algorithm': algo,
            'SNR Improvement (dB)': algo_data['snr_improvement_db'].mean(),
            'PSNR (dB)': algo_data['psnr_db'].mean(),
            'MSE': algo_data['mse'].mean()
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    best_snr_idx = summary_df['SNR Improvement (dB)'].idxmax()
    best_algo = summary_df.loc[best_snr_idx, 'Algorithm']
    best_snr_imp = summary_df.loc[best_snr_idx, 'SNR Improvement (dB)']

    print("\n" + "-" * 50)
    print(f"CONCLUSION: The {best_algo} algorithm achieved the best")
    print(f"performance with an average SNR improvement of {best_snr_imp:.2f} dB.")
    print("=" * 70 + "\n")

