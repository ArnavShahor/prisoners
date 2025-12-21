#!/usr/bin/env python3
"""
Cluster-Based Generosity Analysis for Ultimatum Game
Analyzes whether people are more generous within clusters vs between clusters.
Tests hypothesis: Within-cluster offers > Between-cluster offers
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import argparse
from collections import defaultdict

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Define clusters based on the job list order
CLUSTERS = {
    'TECH': {
        'name': 'Software Engineering & Systems Development',
        'jobs': [
            'Backend Software Engineer',
            'Frontend Software Engineer',
            'Full Stack Software Engineer',
            'Senior Software Developer',
            'Platform Software Engineer'
        ]
    },
    'DATA': {
        'name': 'Data, Analytics & Quantitative Research',
        'jobs': [
            'Data Scientist',
            'Machine Learning Engineer',
            'Data Analyst',
            'Applied Research Scientist',
            'Quantitative Analyst'
        ]
    },
    'HEALTH': {
        'name': 'Healthcare & Clinical Professions',
        'jobs': [
            'Registered Nurse',
            'Clinical Nurse Specialist',
            'Hospital Pharmacist',
            'Physical Therapist',
            'Medical Laboratory Scientist'
        ]
    },
    'SOCIAL': {
        'name': 'Education, Social Services & Public Sector',
        'jobs': [
            'High School Mathematics Teacher',
            'Elementary School Teacher',
            'Academic Counselor',
            'Social Services Case Manager',
            'Educational Program Coordinator'
        ]
    },
    'TRADES': {
        'name': 'Skilled Trades & Engineering-in-the-Physical-World',
        'jobs': [
            'Industrial Electrician',
            'Construction Site Supervisor',
            'Mechanical Maintenance Technician',
            'HVAC Systems Technician',
            'Civil Infrastructure Technician'
        ]
    },
    'CREATIVE': {
        'name': 'Creative, Marketing & Content Roles',
        'jobs': [
            'Content Marketing Manager',
            'Digital Marketing Specialist',
            'Brand Strategist',
            'Creative Content Writer',
            'Visual Communication Designer'
        ]
    }
}


def load_clustered_jobs(filepath: str = "clustered_jobs.json") -> dict:
    """Load clustered jobs and create job-to-cluster mapping."""
    with open(filepath, 'r', encoding='utf-8') as f:
        jobs_list = json.load(f)
    
    # Create mapping from job to cluster
    job_to_cluster = {}
    cluster_to_jobs = defaultdict(list)
    
    current_cluster_idx = 0
    cluster_keys = list(CLUSTERS.keys())
    
    for job in jobs_list:
        # Find which cluster this job belongs to
        found = False
        for cluster_key, cluster_info in CLUSTERS.items():
            if job in cluster_info['jobs']:
                job_to_cluster[job] = cluster_key
                cluster_to_jobs[cluster_key].append(job)
                found = True
                break
        
        if not found:
            # Try to match by partial name or assign to nearest cluster
            print(f"⚠️  Warning: Job '{job}' not found in cluster definitions")
            # Try fuzzy matching or assign to first cluster as fallback
            job_to_cluster[job] = cluster_keys[0]
    
    return job_to_cluster, cluster_to_jobs


def get_cluster_for_job(job: str, job_to_cluster: dict) -> str:
    """Get cluster ID for a job."""
    job = str(job).strip()
    return job_to_cluster.get(job, None)


def load_and_prepare_data(csv_path: str, job_to_cluster: dict) -> pd.DataFrame:
    """Load CSV and add cluster information."""
    df = pd.read_csv(csv_path)
    
    # Add cluster columns
    df['proposer_cluster'] = df['proposer_job'].apply(
        lambda x: get_cluster_for_job(x, job_to_cluster)
    )
    df['responder_cluster'] = df['responder_job'].apply(
        lambda x: get_cluster_for_job(x, job_to_cluster)
    )
    
    # Determine if within-cluster or between-cluster
    df['within_cluster'] = (
        (df['proposer_cluster'].notna()) & 
        (df['responder_cluster'].notna()) &
        (df['proposer_cluster'] == df['responder_cluster'])
    )
    df['between_cluster'] = (
        (df['proposer_cluster'].notna()) & 
        (df['responder_cluster'].notna()) &
        (df['proposer_cluster'] != df['responder_cluster'])
    )
    
    # Create cluster pair identifier
    df['cluster_pair'] = df.apply(
        lambda row: f"{row['proposer_cluster']}-{row['responder_cluster']}" 
        if pd.notna(row['proposer_cluster']) and pd.notna(row['responder_cluster'])
        else None,
        axis=1
    )
    
    # Filter to rows with valid cluster data
    df_clean = df[df['within_cluster'] | df['between_cluster']].copy()
    
    # Print info about missing data
    total_games = len(df)
    games_with_clusters = len(df_clean)
    missing_count = total_games - games_with_clusters
    
    if missing_count > 0:
        print(f"\n[WARNING] {missing_count} out of {total_games} games missing cluster data")
        missing_jobs = set()
        for _, row in df[~df['within_cluster'] & ~df['between_cluster']].iterrows():
            if pd.isna(row['proposer_cluster']):
                missing_jobs.add(row['proposer_job'])
            if pd.isna(row['responder_cluster']):
                missing_jobs.add(row['responder_job'])
        if missing_jobs:
            print(f"   Missing jobs in cluster definitions: {sorted(missing_jobs)}")
        print(f"   Analyzing {games_with_clusters} games with cluster data\n")
    
    # Average offers for games between the same two people
    if 'proposer_name' in df_clean.columns and 'responder_name' in df_clean.columns:
        group_cols = ['proposer_name', 'responder_name']
    elif 'proposer_idx' in df_clean.columns and 'responder_idx' in df_clean.columns:
        group_cols = ['proposer_idx', 'responder_idx']
    else:
        print("⚠️  Warning: Could not find proposer/responder identifiers for averaging")
        return df_clean
    
    # Count games per pair before averaging
    games_per_pair = df_clean.groupby(group_cols).size()
    duplicate_pairs = games_per_pair[games_per_pair > 1]
    
    if len(duplicate_pairs) > 0:
        print(f"📊 Averaging offers for {len(duplicate_pairs)} pairs with multiple games")
        print(f"   Total games before averaging: {len(df_clean)}")
        
        # Average offers and other numeric columns for duplicate pairs
        numeric_cols = ['offer']  # Add other numeric columns if needed
        agg_dict = {col: 'mean' for col in numeric_cols if col in df_clean.columns}
        agg_dict.update({col: 'first' for col in df_clean.columns if col not in numeric_cols})
        
        df_avg = df_clean.groupby(group_cols, as_index=False).agg(agg_dict)
        
        print(f"   Total games after averaging: {len(df_avg)}")
        print(f"   Reduced by {len(df_clean) - len(df_avg)} duplicate games\n")
        
        return df_avg
    
    return df_clean


def calculate_statistics(df: pd.DataFrame):
    """Calculate and print cluster-based statistics."""
    print("=" * 70)
    print("CLUSTER-BASED GENEROSITY ANALYSIS")
    print("=" * 70)
    
    # Basic counts
    within_cluster_games = df[df['within_cluster']]
    between_cluster_games = df[df['between_cluster']]
    
    print(f"\nDataset Overview:")
    print(f"  Total games analyzed: {len(df)}")
    print(f"  Within-cluster games: {len(within_cluster_games)}")
    print(f"  Between-cluster games: {len(between_cluster_games)}")
    
    # Offer statistics
    print(f"\n{'='*70}")
    print("OFFER AMOUNT COMPARISON")
    print(f"{'='*70}")
    
    within_offers = within_cluster_games['offer'].values
    between_offers = between_cluster_games['offer'].values
    
    print(f"\nWithin-Cluster Offers:")
    print(f"  Mean: ${np.mean(within_offers):.2f}")
    print(f"  Median: ${np.median(within_offers):.2f}")
    print(f"  Std Dev: ${np.std(within_offers):.2f}")
    print(f"  Min: ${np.min(within_offers):.2f}")
    print(f"  Max: ${np.max(within_offers):.2f}")
    print(f"  Count: {len(within_offers)}")
    
    print(f"\nBetween-Cluster Offers:")
    print(f"  Mean: ${np.mean(between_offers):.2f}")
    print(f"  Median: ${np.median(between_offers):.2f}")
    print(f"  Std Dev: ${np.std(between_offers):.2f}")
    print(f"  Min: ${np.min(between_offers):.2f}")
    print(f"  Max: ${np.max(between_offers):.2f}")
    print(f"  Count: {len(between_offers)}")
    
    # Statistical test
    print(f"\n{'='*70}")
    print("STATISTICAL TEST: Within-Cluster vs Between-Cluster")
    print(f"{'='*70}")
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(within_offers, between_offers)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(within_offers) - 1) * np.var(within_offers) + 
         (len(between_offers) - 1) * np.var(between_offers)) / 
        (len(within_offers) + len(between_offers) - 2)
    )
    cohens_d = (np.mean(within_offers) - np.mean(between_offers)) / pooled_std
    
    print(f"\nHypothesis: Within-cluster offers > Between-cluster offers")
    print(f"\nResults:")
    print(f"  Mean difference: ${np.mean(within_offers) - np.mean(between_offers):.2f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_size_desc = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size_desc = "small"
    elif abs(cohens_d) < 0.8:
        effect_size_desc = "medium"
    else:
        effect_size_desc = "large"
    
    print(f"  Effect size interpretation: {effect_size_desc}")
    
    # Significance interpretation
    if p_value < 0.001:
        sig_text = "*** (highly significant)"
    elif p_value < 0.01:
        sig_text = "** (very significant)"
    elif p_value < 0.05:
        sig_text = "* (significant)"
    else:
        sig_text = "(not significant)"
    
    print(f"  Significance: {sig_text}")
    
    if p_value < 0.05:
        if np.mean(within_offers) > np.mean(between_offers):
            print(f"\n[SUPPORT] People ARE more generous within clusters!")
        else:
            print(f"\n[REJECT] People are LESS generous within clusters")
    else:
        print(f"\n[INCONCLUSIVE] Cannot reject null hypothesis (no significant difference)")
    
    # Per-cluster analysis
    print(f"\n{'='*70}")
    print("PER-CLUSTER ANALYSIS")
    print(f"{'='*70}")
    
    for cluster_key, cluster_info in CLUSTERS.items():
        cluster_games = df[df['proposer_cluster'] == cluster_key]
        if len(cluster_games) > 0:
            within = cluster_games[cluster_games['responder_cluster'] == cluster_key]
            between = cluster_games[cluster_games['responder_cluster'] != cluster_key]
            
            if len(within) > 0 and len(between) > 0:
                print(f"\n{cluster_info['name']} ({cluster_key}):")
                print(f"  Within-cluster offers: ${np.mean(within['offer']):.2f} (n={len(within)})")
                print(f"  Between-cluster offers: ${np.mean(between['offer']):.2f} (n={len(between)})")
                print(f"  Difference: ${np.mean(within['offer']) - np.mean(between['offer']):.2f}")
                
                # Test for this cluster
                t_cluster, p_cluster = stats.ttest_ind(within['offer'], between['offer'])
                sig_cluster = "***" if p_cluster < 0.001 else "**" if p_cluster < 0.01 else "*" if p_cluster < 0.05 else ""
                print(f"  p-value: {p_cluster:.4f} {sig_cluster}")
    
    print(f"\n{'='*70}\n")


def analyze_cluster_pair_interactions(df: pd.DataFrame):
    """Analyze generosity between specific cluster pairs."""
    print("=" * 70)
    print("CLUSTER-PAIR INTERACTION ANALYSIS")
    print("=" * 70)
    
    # Create matrix of cluster pairs
    cluster_keys = list(CLUSTERS.keys())
    pair_stats = []
    
    for proposer_cluster in cluster_keys:
        for responder_cluster in cluster_keys:
            pair_games = df[
                (df['proposer_cluster'] == proposer_cluster) & 
                (df['responder_cluster'] == responder_cluster)
            ]
            
            if len(pair_games) > 0:
                mean_offer = np.mean(pair_games['offer'])
                pair_stats.append({
                    'proposer': proposer_cluster,
                    'responder': responder_cluster,
                    'mean_offer': mean_offer,
                    'count': len(pair_games),
                    'is_within': proposer_cluster == responder_cluster
                })
    
    pair_df = pd.DataFrame(pair_stats)
    
    # Find most and least generous pairs
    print("\nMost Generous Cluster Pairs:")
    top_pairs = pair_df.nlargest(10, 'mean_offer')
    for idx, row in top_pairs.iterrows():
        within_marker = " [WITHIN]" if row['is_within'] else ""
        print(f"  {row['proposer']} -> {row['responder']}: ${row['mean_offer']:.2f} (n={row['count']}){within_marker}")
    
    print("\nLeast Generous Cluster Pairs:")
    bottom_pairs = pair_df.nsmallest(10, 'mean_offer')
    for idx, row in bottom_pairs.iterrows():
        within_marker = " [WITHIN]" if row['is_within'] else ""
        print(f"  {row['proposer']} -> {row['responder']}: ${row['mean_offer']:.2f} (n={row['count']}){within_marker}")
    
    # Analyze CREATIVE anomaly
    print("\n" + "=" * 70)
    print("CREATIVE CLUSTER ANOMALY INVESTIGATION")
    print("=" * 70)
    
    creative_proposer = df[df['proposer_cluster'] == 'CREATIVE']
    creative_responder = df[df['responder_cluster'] == 'CREATIVE']
    
    if len(creative_proposer) > 0:
        creative_within = creative_proposer[creative_proposer['responder_cluster'] == 'CREATIVE']
        creative_between = creative_proposer[creative_proposer['responder_cluster'] != 'CREATIVE']
        
        print(f"\nCREATIVE as Proposer:")
        if len(creative_within) > 0:
            print(f"  To CREATIVE (within): ${np.mean(creative_within['offer']):.2f} (n={len(creative_within)})")
        if len(creative_between) > 0:
            print(f"  To other clusters: ${np.mean(creative_between['offer']):.2f} (n={len(creative_between)})")
            # Show breakdown by target cluster
            for target_cluster in cluster_keys:
                if target_cluster != 'CREATIVE':
                    target_games = creative_between[creative_between['responder_cluster'] == target_cluster]
                    if len(target_games) > 0:
                        print(f"    -> {target_cluster}: ${np.mean(target_games['offer']):.2f} (n={len(target_games)})")
    
    if len(creative_responder) > 0:
        creative_received_within = creative_responder[creative_responder['proposer_cluster'] == 'CREATIVE']
        creative_received_between = creative_responder[creative_responder['proposer_cluster'] != 'CREATIVE']
        
        print(f"\nCREATIVE as Responder (receiving offers):")
        if len(creative_received_within) > 0:
            print(f"  From CREATIVE (within): ${np.mean(creative_received_within['offer']):.2f} (n={len(creative_received_within)})")
        if len(creative_received_between) > 0:
            print(f"  From other clusters: ${np.mean(creative_received_between['offer']):.2f} (n={len(creative_received_between)})")
            # Show breakdown by source cluster
            for source_cluster in cluster_keys:
                if source_cluster != 'CREATIVE':
                    source_games = creative_received_between[creative_received_between['proposer_cluster'] == source_cluster]
                    if len(source_games) > 0:
                        print(f"    <- {source_cluster}: ${np.mean(source_games['offer']):.2f} (n={len(source_games)})")
    
    print(f"\n{'='*70}\n")
    
    return pair_df


def analyze_directional_effects(df: pd.DataFrame):
    """Analyze if clusters are more generous as proposers vs responders."""
    print("=" * 70)
    print("DIRECTIONAL ANALYSIS: Proposer vs Responder Generosity")
    print("=" * 70)
    
    cluster_keys = list(CLUSTERS.keys())
    
    # For each cluster, compare their generosity as proposer vs what they receive as responder
    directional_stats = []
    
    for cluster_key in cluster_keys:
        # As proposer (giving offers)
        as_proposer = df[df['proposer_cluster'] == cluster_key]
        # As responder (receiving offers)
        as_responder = df[df['responder_cluster'] == cluster_key]
        
        if len(as_proposer) > 0 and len(as_responder) > 0:
            proposer_mean = np.mean(as_proposer['offer'])
            responder_mean = np.mean(as_responder['offer'])
            difference = proposer_mean - responder_mean
            
            directional_stats.append({
                'cluster': cluster_key,
                'as_proposer_mean': proposer_mean,
                'as_responder_mean': responder_mean,
                'difference': difference,
                'proposer_count': len(as_proposer),
                'responder_count': len(as_responder)
            })
    
    if directional_stats:
        dir_df = pd.DataFrame(directional_stats)
        
        print("\nCluster Generosity: Giving vs Receiving")
        print(f"{'Cluster':<10} {'As Proposer':<15} {'As Responder':<15} {'Difference':<12} {'Interpretation'}")
        print("-" * 80)
        
        for idx, row in dir_df.iterrows():
            cluster_name = CLUSTERS[row['cluster']]['name'][:9] if len(CLUSTERS[row['cluster']]['name']) > 9 else CLUSTERS[row['cluster']]['name']
            diff = row['difference']
            if diff > 1.0:
                interpretation = "More generous when giving"
            elif diff < -1.0:
                interpretation = "Receive more generous offers"
            else:
                interpretation = "Balanced"
            
            print(f"{row['cluster']:<10} ${row['as_proposer_mean']:>6.2f} (n={row['proposer_count']:<3}) "
                  f"${row['as_responder_mean']:>6.2f} (n={row['responder_count']:<3}) "
                  f"${diff:>6.2f}      {interpretation}")
        
        print(f"\n{'='*70}\n")
    
    return dir_df if directional_stats else None


def create_visualizations(df: pd.DataFrame, output_dir: str = "visualizations", csv_name: str = ""):
    """Create visualizations for cluster-based analysis."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    within_offers = df[df['within_cluster']]['offer'].values
    between_offers = df[df['between_cluster']]['offer'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cluster-Based Generosity Analysis: Within vs Between Clusters', 
                 fontsize=16, fontweight='bold')
    
    # 1. Box plot comparison
    ax1 = axes[0, 0]
    box_data = [within_offers, between_offers]
    bp = ax1.boxplot(box_data, labels=['Within Cluster', 'Between Clusters'], 
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax1.set_ylabel('Offer Amount ($)', fontsize=12)
    ax1.set_title('Offer Distribution: Within vs Between Clusters', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines
    ax1.axhline(np.mean(within_offers), color='blue', linestyle='--', alpha=0.7, label=f'Within mean: ${np.mean(within_offers):.2f}')
    ax1.axhline(np.mean(between_offers), color='red', linestyle='--', alpha=0.7, label=f'Between mean: ${np.mean(between_offers):.2f}')
    ax1.legend()
    
    # 2. Violin plot
    ax2 = axes[0, 1]
    violin_data = pd.DataFrame({
        'Offer': np.concatenate([within_offers, between_offers]),
        'Type': ['Within Cluster'] * len(within_offers) + ['Between Clusters'] * len(between_offers)
    })
    sns.violinplot(data=violin_data, x='Type', y='Offer', ax=ax2, palette=['lightblue', 'lightcoral'])
    ax2.set_ylabel('Offer Amount ($)', fontsize=12)
    ax2.set_title('Offer Distribution (Violin Plot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Histogram comparison
    ax3 = axes[1, 0]
    ax3.hist(within_offers, bins=20, alpha=0.6, label='Within Cluster', color='lightblue', edgecolor='black')
    ax3.hist(between_offers, bins=20, alpha=0.6, label='Between Clusters', color='lightcoral', edgecolor='black')
    ax3.set_xlabel('Offer Amount ($)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Offer Distribution Histogram', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-cluster comparison
    ax4 = axes[1, 1]
    cluster_stats = []
    for cluster_key, cluster_info in CLUSTERS.items():
        cluster_games = df[df['proposer_cluster'] == cluster_key]
        if len(cluster_games) > 0:
            within = cluster_games[cluster_games['responder_cluster'] == cluster_key]
            between = cluster_games[cluster_games['responder_cluster'] != cluster_key]
            if len(within) > 0 and len(between) > 0:
                cluster_stats.append({
                    'Cluster': cluster_key,
                    'Within': np.mean(within['offer']),
                    'Between': np.mean(between['offer'])
                })
    
    if cluster_stats:
        cluster_df = pd.DataFrame(cluster_stats)
        x = np.arange(len(cluster_df))
        width = 0.35
        ax4.bar(x - width/2, cluster_df['Within'], width, label='Within Cluster', color='lightblue')
        ax4.bar(x + width/2, cluster_df['Between'], width, label='Between Clusters', color='lightcoral')
        ax4.set_xlabel('Cluster', fontsize=12)
        ax4.set_ylabel('Mean Offer ($)', fontsize=12)
        ax4.set_title('Mean Offers by Cluster', fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(cluster_df['Cluster'], rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_dir) / f"cluster_generosity_analysis_{csv_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization: {output_file}")
    plt.close()


def create_cluster_pair_heatmap(df: pd.DataFrame, pair_df: pd.DataFrame, output_dir: str = "visualizations", csv_name: str = ""):
    """Create heatmap showing generosity between all cluster pairs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    cluster_keys = list(CLUSTERS.keys())
    
    # Create matrix
    heatmap_matrix = np.full((len(cluster_keys), len(cluster_keys)), np.nan)
    
    for i, proposer in enumerate(cluster_keys):
        for j, responder in enumerate(cluster_keys):
            pair_data = pair_df[
                (pair_df['proposer'] == proposer) & 
                (pair_df['responder'] == responder)
            ]
            if len(pair_data) > 0:
                heatmap_matrix[i, j] = pair_data.iloc[0]['mean_offer']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(heatmap_matrix, cmap='RdYlGn', aspect='auto', vmin=35, vmax=45)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(cluster_keys)))
    ax.set_yticks(np.arange(len(cluster_keys)))
    ax.set_xticklabels(cluster_keys, rotation=45, ha='right')
    ax.set_yticklabels(cluster_keys)
    
    # Add text annotations
    for i in range(len(cluster_keys)):
        for j in range(len(cluster_keys)):
            if not np.isnan(heatmap_matrix[i, j]):
                text = ax.text(j, i, f'${heatmap_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontweight='bold')
                # Highlight within-cluster pairs
                if i == j:
                    rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                             edgecolor='blue', linewidth=3)
                    ax.add_patch(rect)
    
    ax.set_xlabel('Responder Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proposer Cluster', fontsize=12, fontweight='bold')
    ax.set_title('Mean Offer Amount by Cluster Pair\n(Blue boxes = within-cluster pairs)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Offer ($)', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / f"cluster_pair_heatmap_{csv_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved cluster-pair heatmap: {output_file}")
    plt.close()


def create_directional_visualization(dir_df: pd.DataFrame, output_dir: str = "visualizations", csv_name: str = ""):
    """Create visualization comparing proposer vs responder generosity by cluster."""
    if dir_df is None or len(dir_df) == 0:
        return
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(dir_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, dir_df['as_proposer_mean'], width, 
                   label='As Proposer (Giving)', color='lightblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, dir_df['as_responder_mean'], width,
                   label='As Responder (Receiving)', color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Offer Amount ($)', fontsize=12, fontweight='bold')
    ax.set_title('Directional Generosity: Giving vs Receiving by Cluster', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dir_df['cluster'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_file = Path(output_dir) / f"directional_generosity_{csv_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved directional analysis: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze cluster-based generosity in ultimatum game results'
    )
    parser.add_argument(
        'csv_file',
        type=str,
        help='Path to CSV file with ultimatum game results'
    )
    parser.add_argument(
        '--clustered-jobs',
        type=str,
        default='clustered_jobs.json',
        help='Path to clustered jobs JSON file (default: clustered_jobs.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations (default: visualizations)'
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.csv_file}")
    print(f"Loading cluster definitions from: {args.clustered_jobs}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load cluster mappings
    job_to_cluster, cluster_to_jobs = load_clustered_jobs(args.clustered_jobs)
    print(f"Loaded {len(job_to_cluster)} jobs across {len(CLUSTERS)} clusters")
    
    # Load and prepare data
    df = load_and_prepare_data(args.csv_file, job_to_cluster)
    
    if len(df) == 0:
        print("[ERROR] No data with valid cluster information found!")
        return
    
    # Extract CSV filename for output naming
    csv_path = Path(args.csv_file)
    csv_name = csv_path.stem
    
    # Calculate statistics
    calculate_statistics(df)
    
    # Cluster-pair interaction analysis
    print("\nAnalyzing cluster-pair interactions...")
    pair_df = analyze_cluster_pair_interactions(df)
    
    # Directional analysis
    print("\nAnalyzing directional effects...")
    dir_df = analyze_directional_effects(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, args.output_dir, csv_name)
    
    # Create cluster-pair heatmap
    if pair_df is not None and len(pair_df) > 0:
        create_cluster_pair_heatmap(df, pair_df, args.output_dir, csv_name)
    
    # Create directional visualization
    if dir_df is not None and len(dir_df) > 0:
        create_directional_visualization(dir_df, args.output_dir, csv_name)
    
    print(f"\n[OK] Analysis complete! All visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

