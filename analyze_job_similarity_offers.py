#!/usr/bin/env python3
"""
Focused Analysis: Job Similarity vs Offer Amount in Ultimatum Game
Analyzes the relationship between job similarity between proposer and responder
and the offer amount made in the ultimatum game.
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
import argparse

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11


def load_job_similarities(filepath: str = "job_similarities.json") -> dict:
    """Load job similarity matrix from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_job_similarity(job1: str, job2: str, similarities: dict) -> float:
    """
    Get similarity score between two jobs.
    Returns None if either job is not in the similarity matrix.
    """
    if pd.isna(job1) or pd.isna(job2):
        return None
    
    job1 = str(job1).strip()
    job2 = str(job2).strip()
    
    if job1 not in similarities:
        return None
    if job2 in similarities[job1]:
        return similarities[job1][job2]
    # Try reverse lookup
    if job2 in similarities and job1 in similarities[job2]:
        return similarities[job2][job1]
    return None


def load_and_prepare_data(csv_path: str, similarities: dict) -> pd.DataFrame:
    """Load CSV and add job similarity scores."""
    df = pd.read_csv(csv_path)
    
    # Add job similarity column
    df['job_similarity'] = df.apply(
        lambda row: get_job_similarity(
            row['proposer_job'], 
            row['responder_job'], 
            similarities
        ),
        axis=1
    )
    
    # Filter to rows with valid similarity scores
    df_clean = df[df['job_similarity'].notna()].copy()
    
    # Print info about missing data
    total_games = len(df)
    games_with_similarity = len(df_clean)
    missing_count = total_games - games_with_similarity
    
    if missing_count > 0:
        print(f"\n⚠️  Warning: {missing_count} out of {total_games} games missing similarity data")
        missing_jobs = set()
        for _, row in df[df['job_similarity'].isna()].iterrows():
            if row['proposer_job'] not in similarities:
                missing_jobs.add(row['proposer_job'])
            if row['responder_job'] not in similarities:
                missing_jobs.add(row['responder_job'])
        if missing_jobs:
            print(f"   Missing jobs in similarity matrix: {sorted(missing_jobs)}")
        print(f"   Analyzing {games_with_similarity} games with similarity data\n")
    
    return df_clean


def calculate_statistics(df: pd.DataFrame):
    """Calculate and print statistical analysis."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: Job Similarity vs Offer Amount")
    print("="*70)
    
    # Basic statistics
    print(f"\nDataset:")
    print(f"  Total games with similarity data: {len(df)}")
    print(f"  Unique proposer-responder job pairs: {len(df.groupby(['proposer_job', 'responder_job']))}")
    
    print(f"\nJob Similarity Statistics:")
    print(f"  Mean: {df['job_similarity'].mean():.4f}")
    print(f"  Median: {df['job_similarity'].median():.4f}")
    print(f"  Std Dev: {df['job_similarity'].std():.4f}")
    print(f"  Min: {df['job_similarity'].min():.4f}")
    print(f"  Max: {df['job_similarity'].max():.4f}")
    
    print(f"\nOffer Amount Statistics:")
    print(f"  Mean: ${df['offer'].mean():.2f}")
    print(f"  Median: ${df['offer'].median():.2f}")
    print(f"  Std Dev: ${df['offer'].std():.2f}")
    print(f"  Min: ${df['offer'].min():.2f}")
    print(f"  Max: ${df['offer'].max():.2f}")
    
    # Correlation analysis
    correlation = df['job_similarity'].corr(df['offer'])
    print(f"\n{'='*70}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Pearson Correlation Coefficient: {correlation:.4f}")
    
    # Statistical significance test
    r, p_value = stats.pearsonr(df['job_similarity'], df['offer'])
    print(f"P-value: {p_value:.6f}")
    
    # Calculate confidence interval for correlation coefficient
    n = len(df)
    z = np.arctanh(r)  # Fisher's z-transformation
    se = 1 / np.sqrt(n - 3)  # Standard error
    z_crit = 1.96  # 95% confidence interval
    z_lower = z - z_crit * se
    z_upper = z + z_crit * se
    r_lower = np.tanh(z_lower)
    r_upper = np.tanh(z_upper)
    
    if p_value < 0.001:
        significance = "*** (highly significant)"
    elif p_value < 0.01:
        significance = "** (very significant)"
    elif p_value < 0.05:
        significance = "* (significant)"
    else:
        significance = "(not significant)"
    
    print(f"Significance: {significance}")
    print(f"95% Confidence Interval for r: [{r_lower:.4f}, {r_upper:.4f}]")
    
    if p_value >= 0.05:
        print(f"\n⚠️  WARNING: The correlation is NOT statistically significant (p ≥ 0.05)")
        print(f"   This means we cannot reject the null hypothesis that r = 0")
        print(f"   The observed correlation could be due to random chance")
    
    # Interpretation
    print(f"\nInterpretation:")
    if abs(correlation) < 0.1:
        print(f"  → Very weak correlation: Job similarity explains very little of the variation in offers")
    elif abs(correlation) < 0.3:
        print(f"  → Weak correlation: Job similarity has a small effect on offer amounts")
    elif abs(correlation) < 0.5:
        print(f"  → Moderate correlation: Job similarity has a noticeable effect on offer amounts")
    elif abs(correlation) < 0.7:
        print(f"  → Strong correlation: Job similarity has a substantial effect on offer amounts")
    else:
        print(f"  → Very strong correlation: Job similarity strongly predicts offer amounts")
    
    if correlation > 0:
        print(f"  → Positive relationship: Higher job similarity is associated with HIGHER offers")
    elif correlation < 0:
        print(f"  → Negative relationship: Higher job similarity is associated with LOWER offers")
    
    # Linear regression
    print(f"\n{'='*70}")
    print(f"LINEAR REGRESSION ANALYSIS")
    print(f"{'='*70}")
    X = df[['job_similarity']].values
    y = df['offer'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    # Calculate standard errors and confidence intervals for regression coefficients
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (len(y) - 2)  # Mean squared error
    var_x = np.var(X, ddof=1)
    se_slope = np.sqrt(mse / (var_x * (len(X) - 1)))
    se_intercept = np.sqrt(mse * (1/len(X) + np.mean(X)**2 / (var_x * (len(X) - 1))))
    
    # t-statistic and p-value for slope
    t_slope = slope / se_slope
    p_slope = 2 * (1 - stats.t.cdf(abs(t_slope), len(X) - 2))
    
    # 95% confidence intervals
    t_crit = stats.t.ppf(0.975, len(X) - 2)
    slope_ci_lower = slope - t_crit * se_slope
    slope_ci_upper = slope + t_crit * se_slope
    intercept_ci_lower = intercept - t_crit * se_intercept
    intercept_ci_upper = intercept + t_crit * se_intercept
    
    print(f"Regression Equation: Offer = {slope:.2f} × Similarity + {intercept:.2f}")
    print(f"\nRegression Coefficient (Slope) Statistics:")
    print(f"  Coefficient (β): {slope:.4f}")
    print(f"  Standard Error: {se_slope:.4f}")
    print(f"  t-statistic: {t_slope:.4f}")
    print(f"  P-value: {p_slope:.6f}")
    print(f"  95% Confidence Interval: [{slope_ci_lower:.4f}, {slope_ci_upper:.4f}]")
    
    if p_slope < 0.001:
        slope_sig = "*** (highly significant)"
    elif p_slope < 0.01:
        slope_sig = "** (very significant)"
    elif p_slope < 0.05:
        slope_sig = "* (significant)"
    else:
        slope_sig = "(not significant)"
    print(f"  Significance: {slope_sig}")
    
    print(f"\nIntercept Statistics:")
    print(f"  Intercept (α): {intercept:.4f}")
    print(f"  Standard Error: {se_intercept:.4f}")
    print(f"  95% Confidence Interval: [{intercept_ci_lower:.4f}, {intercept_ci_upper:.4f}]")
    
    print(f"\nR² (Coefficient of Determination): {r_squared:.4f}")
    print(f"  → This means job similarity explains {r_squared*100:.2f}% of the variance in offer amounts")
    
    if p_slope >= 0.05:
        print(f"\n⚠️  WARNING: The regression coefficient is NOT statistically significant (p ≥ 0.05)")
        print(f"   We cannot conclude that job similarity has a meaningful effect on offer amounts")
    
    # Predictions at different similarity levels
    print(f"\nPredicted Offers at Different Similarity Levels:")
    similarity_levels = [0.2, 0.4, 0.6, 0.8]
    for sim in similarity_levels:
        predicted_offer = slope * sim + intercept
        print(f"  Similarity {sim:.1f}: ${predicted_offer:.2f}")
    
    print("="*70 + "\n")


def create_visualizations(df: pd.DataFrame, output_dir: str = "visualizations", csv_name: str = ""):
    """Create focused, intuitive visualizations of job similarity vs offer amount."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create filename suffix from CSV name
    if csv_name:
        file_suffix = f"_{csv_name}"
    else:
        file_suffix = ""
    
    # Calculate statistics for annotations
    correlation = df['job_similarity'].corr(df['offer'])
    r, p_value = stats.pearsonr(df['job_similarity'], df['offer'])
    X = df[['job_similarity']].values
    y = df['offer'].values
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    # Create figure with a cleaner 2x2 layout
    fig = plt.figure(figsize=(16, 12))
    # Adjust grid layout for better alignment
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, 
                          left=0.08, right=0.97, top=0.94, bottom=0.08)
    
    # ========== PLOT 1: Main Scatter Plot with Regression Line ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create scatter plot with color mapping
    scatter = ax1.scatter(
        df['job_similarity'], 
        df['offer'],
        alpha=0.6,
        s=50,
        edgecolors='darkblue',
        linewidth=1,
        c=df['job_similarity'],
        cmap='plasma',
        zorder=3
    )
    
    # Add regression line
    z = np.polyfit(df['job_similarity'], df['offer'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['job_similarity'].min(), df['job_similarity'].max(), 100)
    y_line = p(x_line)
    ax1.plot(x_line, y_line, "r-", linewidth=3, alpha=0.9, 
             label=f'Trend Line: Offer = {slope:.1f} × Similarity + {intercept:.1f}', zorder=4)
    
    # Add shaded confidence region
    y_std = df['offer'].std()
    ax1.fill_between(x_line, y_line - y_std, y_line + y_std, 
                     alpha=0.2, color='red', label='±1 Std Dev')
    
    # Add correlation and significance info in a clear box
    sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    info_text = f'Correlation: r = {correlation:.3f} {sig_text}\n'
    info_text += f'R² = {r_squared:.3f} ({r_squared*100:.1f}% variance explained)\n'
    info_text += f'p-value = {p_value:.4f}'
    
    # Add correlation info box - positioned to avoid legend
    ax1.text(0.02, 0.98, info_text,
             transform=ax1.transAxes,
             fontsize=11,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                      linewidth=2, alpha=0.95, pad=8),
             family='monospace',
             zorder=5)
    
    # Add interpretation text
    if correlation > 0.1:
        interp = "Higher job similarity -> Higher offers"
        color = 'green'
    elif correlation < -0.1:
        interp = "Higher job similarity -> Lower offers"
        color = 'red'
    else:
        interp = "Weak relationship between similarity and offers"
        color = 'gray'
    
    ax1.text(0.98, 0.05, interp,
             transform=ax1.transAxes,
             fontsize=12,
             fontweight='bold',
             color=color,
             horizontalalignment='right',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      edgecolor=color, linewidth=2, alpha=0.9, pad=8),
             zorder=5)
    
    ax1.set_xlabel('Job Similarity Score (0 = Very Different, 1 = Very Similar)', 
                   fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Offer Amount ($)', fontsize=14, fontweight='bold', labelpad=10)
    ax1.set_title('Relationship Between Job Similarity and Offer Amount', 
                  fontsize=16, fontweight='bold', pad=15, loc='left')
    # Position legend better - avoid overlap with data and text boxes
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.95, 
              edgecolor='black', frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.98, 0.85))
    ax1.grid(True, alpha=0.3, linestyle='--', zorder=1)
    ax1.set_ylim([df['offer'].min() - 2, df['offer'].max() + 2])
    
    # Add colorbar with proper spacing
    cbar = plt.colorbar(scatter, ax=ax1, pad=0.02, aspect=30)
    cbar.set_label('Job Similarity Score', fontsize=12, fontweight='bold', labelpad=10)
    
    # ========== PLOT 2: Grouped Comparison ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create meaningful similarity groups
    similarity_mean = df['job_similarity'].median()
    df['similarity_group'] = df['job_similarity'].apply(
        lambda x: 'Below Median\n(Less Similar)' if x < similarity_mean 
        else 'Above Median\n(More Similar)'
    )
    
    # Create grouped box plot
    groups = df['similarity_group'].unique()
    data_to_plot = [df[df['similarity_group'] == group]['offer'].values for group in groups]
    
    bp = ax2.boxplot(data_to_plot, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True)
    ax2.set_xticklabels(groups)
    
    # Color the boxes
    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(2)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    # Add mean values on top
    for i, group in enumerate(groups):
        mean_val = df[df['similarity_group'] == group]['offer'].mean()
        count = len(df[df['similarity_group'] == group])
        ax2.text(i+1, mean_val + 1.5, f'${mean_val:.1f}\n(n={count})',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=4))
    
    ax2.set_ylabel('Offer Amount ($)', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_xlabel('Job Similarity Group', fontsize=12, fontweight='bold', labelpad=8)
    ax2.set_title('Offers: Less Similar vs More Similar Jobs', 
                  fontsize=13, fontweight='bold', pad=10, loc='left')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim([df['offer'].min() - 2, df['offer'].max() + 4])
    
    # ========== PLOT 3: Average Offer by Similarity Bins ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Create bins that make sense for the data range
    similarity_min = df['job_similarity'].min()
    similarity_max = df['job_similarity'].max()
    similarity_range = similarity_max - similarity_min
    
    # Use 3 bins based on actual data range
    if similarity_range > 0.2:
        num_bins = 3
        bin_edges = np.linspace(similarity_min, similarity_max, num_bins + 1)
        labels = [f'Low\n({bin_edges[i]:.2f}-{bin_edges[i+1]:.2f})' 
                 for i in range(num_bins)]
    else:
        # If range is small, use 2 bins
        num_bins = 2
        bin_edges = np.linspace(similarity_min, similarity_max, num_bins + 1)
        labels = [f'Low\n({bin_edges[0]:.2f}-{bin_edges[1]:.2f})',
                 f'High\n({bin_edges[1]:.2f}-{bin_edges[2]:.2f})']
    
    df['similarity_bin'] = pd.cut(df['job_similarity'], bins=bin_edges, labels=labels)
    
    avg_by_bin = df.groupby('similarity_bin', observed=True)['offer'].agg(['mean', 'std', 'count'])
    
    if len(avg_by_bin) > 0:
        x_pos = np.arange(len(avg_by_bin))
        colors_bar = plt.cm.plasma(np.linspace(0.3, 0.9, len(avg_by_bin)))
        
        bars = ax3.bar(x_pos, avg_by_bin['mean'], yerr=avg_by_bin['std'], 
                      capsize=8, alpha=0.8, edgecolor='black', linewidth=2,
                      color=colors_bar)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(avg_by_bin.index, fontsize=11, fontweight='bold', ha='center')
        ax3.set_ylabel('Average Offer Amount ($)', fontsize=12, fontweight='bold', labelpad=10)
        ax3.set_title('Average Offer by Similarity Level', 
                      fontsize=13, fontweight='bold', pad=12, loc='left')
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add value labels on bars
        for i, (bar, mean_val, std_val, count) in enumerate(zip(bars, avg_by_bin['mean'], 
                                                                 avg_by_bin['std'], 
                                                                 avg_by_bin['count'])):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                    f'${mean_val:.1f}\n(n={int(count)})',
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=4))
        
        ax3.set_ylim([0, df['offer'].max() + 6])
        ax3.set_xlabel('Similarity Level', fontsize=12, fontweight='bold', labelpad=8)
    
    # Add overall title - centered and properly positioned
    fig.suptitle('Job Similarity vs Offer Amount in Ultimatum Game', 
                 fontsize=17, fontweight='bold', y=0.98, x=0.5, ha='center')
    
    # Save the figure
    output_file = output_path / f'job_similarity_vs_offer{file_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved visualization: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze the relationship between job similarity and offer amounts in ultimatum game'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file with ultimatum game results'
    )
    parser.add_argument(
        '--similarities',
        type=str,
        default='job_similarities.json',
        help='Path to job similarities JSON file (default: job_similarities.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations (default: visualizations)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.csv_file}")
    print(f"Loading similarities from: {args.similarities}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load data
    similarities = load_job_similarities(args.similarities)
    df = load_and_prepare_data(args.csv_file, similarities)
    
    if len(df) == 0:
        print("❌ Error: No data with valid job similarity scores found!")
        print("   Check that job names in CSV match those in similarity matrix.")
        return
    
    # Calculate and print statistics
    calculate_statistics(df)
    
    # Create visualizations
    print("Generating visualization...")
    create_visualizations(df, args.output_dir)
    
    print(f"\n✅ Analysis complete! Visualization saved to: {args.output_dir}/")


def create_gender_visualizations(df: pd.DataFrame, output_dir: str = "visualizations", csv_name: str = ""):
    """Create 4 visualizations comparing gender combinations: M-M, F-F, M-F, F-M."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create filename suffix from CSV name
    if csv_name:
        file_suffix = f"_{csv_name}"
    else:
        file_suffix = ""
    
    # Define gender combinations
    gender_combinations = [
        ('Male', 'Male', 'Men vs Men', 'M-M'),
        ('Female', 'Female', 'Women vs Women', 'F-F'),
        ('Male', 'Female', 'Men vs Women', 'M-F'),
        ('Female', 'Male', 'Women vs Men', 'F-M')
    ]
    
    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Job Similarity vs Offer Amount by Gender Combination', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    for idx, (prop_gender, resp_gender, title, label) in enumerate(gender_combinations):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Filter data for this gender combination
        df_gender = df[
            (df['proposer_gender'] == prop_gender) & 
            (df['responder_gender'] == resp_gender)
        ].copy()
        
        if len(df_gender) == 0:
            ax.text(0.5, 0.5, f'No data available\nfor {title}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color='gray')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Job Similarity Score', fontsize=11)
            ax.set_ylabel('Offer Amount ($)', fontsize=11)
            continue
        
        # Create scatter plot
        scatter = ax.scatter(
            df_gender['job_similarity'], 
            df_gender['offer'],
            alpha=0.6,
            s=60,
            edgecolors='darkblue',
            linewidth=1,
            c=df_gender['job_similarity'],
            cmap='plasma',
            zorder=3
        )
        
        # Calculate statistics
        correlation = df_gender['job_similarity'].corr(df_gender['offer'])
        r, p_value = stats.pearsonr(df_gender['job_similarity'], df_gender['offer'])
        
        # Linear regression
        X = df_gender[['job_similarity']].values
        y = df_gender['offer'].values
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(X, y)
        
        # Add regression line
        x_line = np.linspace(df_gender['job_similarity'].min(), df_gender['job_similarity'].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r-", linewidth=2.5, alpha=0.9, 
               label=f'y = {slope:.1f}x + {intercept:.1f}', zorder=4)
        
        # Add shaded confidence region
        y_std = df_gender['offer'].std()
        ax.fill_between(x_line, y_line - y_std, y_line + y_std, 
                        alpha=0.15, color='red', zorder=2)
        
        # Add statistics text box
        sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        info_text = f'n = {len(df_gender)}\n'
        info_text += f'r = {correlation:.3f} {sig_text}\n'
        info_text += f'R² = {r_squared:.3f}\n'
        info_text += f'p = {p_value:.4f}'
        
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                        linewidth=1.5, alpha=0.95, pad=6),
               family='monospace',
               zorder=5)
        
        # Add mean offer annotation
        mean_offer = df_gender['offer'].mean()
        ax.axhline(mean_offer, color='green', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label=f'Mean: ${mean_offer:.1f}', zorder=1)
        
        # Styling
        ax.set_xlabel('Job Similarity Score', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_ylabel('Offer Amount ($)', fontsize=12, fontweight='bold', labelpad=8)
        ax.set_title(f'{title} ({label})', fontsize=14, fontweight='bold', pad=12)
        ax.legend(fontsize=9, loc='best', framealpha=0.9, edgecolor='black', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.set_ylim([df_gender['offer'].min() - 2, df_gender['offer'].max() + 3])
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.01, aspect=25)
        cbar.set_label('Similarity', fontsize=9, labelpad=5)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    output_file = output_path / f'gender_analysis{file_suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved gender visualization: {output_file}")
    plt.close()
    
    # Print gender-specific statistics
    print("\n" + "="*70)
    print("GENDER-SPECIFIC STATISTICS")
    print("="*70)
    
    for prop_gender, resp_gender, title, label in gender_combinations:
        df_gender = df[
            (df['proposer_gender'] == prop_gender) & 
            (df['responder_gender'] == resp_gender)
        ].copy()
        
        if len(df_gender) > 0:
            correlation = df_gender['job_similarity'].corr(df_gender['offer'])
            r, p_value = stats.pearsonr(df_gender['job_similarity'], df_gender['offer'])
            mean_offer = df_gender['offer'].mean()
            std_offer = df_gender['offer'].std()
            
            # Calculate confidence interval for correlation
            n_gender = len(df_gender)
            z_gender = np.arctanh(correlation)
            se_gender = 1 / np.sqrt(n_gender - 3)
            z_lower_gender = z_gender - 1.96 * se_gender
            z_upper_gender = z_gender + 1.96 * se_gender
            r_lower_gender = np.tanh(z_lower_gender)
            r_upper_gender = np.tanh(z_upper_gender)
            
            print(f"\n{title} ({label}):")
            print(f"  Games: {len(df_gender)}")
            print(f"  Mean Offer: ${mean_offer:.2f} (std: ${std_offer:.2f})")
            print(f"  Correlation: r = {correlation:.4f}, p = {p_value:.4f}")
            print(f"  95% CI for r: [{r_lower_gender:.4f}, {r_upper_gender:.4f}]")
            
            sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "(not significant)"
            print(f"  Significance: {sig_text}")
            
            if p_value >= 0.05:
                print(f"  ⚠️  NOT statistically significant - cannot reject H₀: r = 0")
        else:
            print(f"\n{title} ({label}): No data available")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze the relationship between job similarity and offer amounts in ultimatum game'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file with ultimatum game results'
    )
    parser.add_argument(
        '--similarities',
        type=str,
        default='job_similarities.json',
        help='Path to job similarities JSON file (default: job_similarities.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations (default: visualizations)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.csv_file}")
    print(f"Loading similarities from: {args.similarities}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load data
    similarities = load_job_similarities(args.similarities)
    df = load_and_prepare_data(args.csv_file, similarities)
    
    if len(df) == 0:
        print("❌ Error: No data with valid job similarity scores found!")
        print("   Check that job names in CSV match those in similarity matrix.")
        return
    
    # Extract CSV filename (without extension) for output naming
    csv_path = Path(args.csv_file)
    csv_name = csv_path.stem  # Gets filename without extension
    
    # Calculate and print statistics
    calculate_statistics(df)
    
    # Create visualizations
    print("Generating visualizations...")
    create_visualizations(df, args.output_dir, csv_name)
    
    # Create gender-based visualizations
    print("Generating gender-based visualizations...")
    create_gender_visualizations(df, args.output_dir, csv_name)
    
    print(f"\n✅ Analysis complete! Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

