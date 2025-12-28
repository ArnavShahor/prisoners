#!/usr/bin/env python3
"""
Linear Models for Cluster Generosity Analysis
Tests hypothesis with various controls and model specifications
Uses sklearn for compatibility
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from scipy import stats
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Define clusters
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
    
    job_to_cluster = {}
    for job in jobs_list:
        for cluster_key, cluster_info in CLUSTERS.items():
            if job in cluster_info['jobs']:
                job_to_cluster[job] = cluster_key
                break
    
    return job_to_cluster


def load_job_similarities(filepath: str = "job_similarities.json") -> dict:
    """Load job similarity matrix."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_job_similarity(job1: str, job2: str, similarities: dict) -> float:
    """Get similarity score between two jobs."""
    if pd.isna(job1) or pd.isna(job2):
        return None
    
    job1 = str(job1).strip()
    job2 = str(job2).strip()
    
    if job1 in similarities:
        if job2 in similarities[job1]:
            return similarities[job1][job2]
    if job2 in similarities:
        if job1 in similarities[job2]:
            return similarities[job2][job1]
    return None


def get_cluster_for_job(job: str, job_to_cluster: dict) -> str:
    """Get cluster ID for a job."""
    job = str(job).strip()
    return job_to_cluster.get(job, None)


def prepare_data(csv_path: str, job_to_cluster: dict, similarities: dict = None) -> pd.DataFrame:
    """Load and prepare data for modeling."""
    df = pd.read_csv(csv_path)
    
    # Add cluster columns
    df['proposer_cluster'] = df['proposer_job'].apply(
        lambda x: get_cluster_for_job(x, job_to_cluster)
    )
    df['responder_cluster'] = df['responder_job'].apply(
        lambda x: get_cluster_for_job(x, job_to_cluster)
    )
    
    # Create within_cluster variable
    df['within_cluster'] = (
        (df['proposer_cluster'].notna()) & 
        (df['responder_cluster'].notna()) &
        (df['proposer_cluster'] == df['responder_cluster'])
    ).astype(int)
    
    # Add job similarity if available
    if similarities:
        df['job_similarity'] = df.apply(
            lambda row: get_job_similarity(
                row['proposer_job'], 
                row['responder_job'], 
                similarities
            ),
            axis=1
        )
    else:
        df['job_similarity'] = None
    
    # Filter to valid data
    df_clean = df[
        (df['proposer_cluster'].notna()) & 
        (df['responder_cluster'].notna()) &
        (df['offer'].notna())
    ].copy()
    
    # Create cluster dummies for proposer (TECH as reference)
    for cluster_key in CLUSTERS.keys():
        if cluster_key != 'TECH':
            df_clean[f'proposer_{cluster_key}'] = (df_clean['proposer_cluster'] == cluster_key).astype(int)
    
    # Create cluster dummies for responder (TECH as reference)
    for cluster_key in CLUSTERS.keys():
        if cluster_key != 'TECH':
            df_clean[f'responder_{cluster_key}'] = (df_clean['responder_cluster'] == cluster_key).astype(int)
    
    # Average for duplicate pairs if needed
    if 'proposer_name' in df_clean.columns and 'responder_name' in df_clean.columns:
        group_cols = ['proposer_name', 'responder_name']
    elif 'proposer_idx' in df_clean.columns and 'responder_idx' in df_clean.columns:
        group_cols = ['proposer_idx', 'responder_idx']
    else:
        return df_clean
    
    games_per_pair = df_clean.groupby(group_cols).size()
    if len(games_per_pair[games_per_pair > 1]) > 0:
        numeric_cols = ['offer', 'job_similarity'] if 'job_similarity' in df_clean.columns else ['offer']
        agg_dict = {col: 'mean' for col in numeric_cols if col in df_clean.columns}
        agg_dict.update({col: 'first' for col in df_clean.columns if col not in numeric_cols})
        df_clean = df_clean.groupby(group_cols, as_index=False).agg(agg_dict)
    
    return df_clean


def fit_linear_model(X, y, feature_names):
    """Fit linear model and return results dictionary."""
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    n, p = X.shape
    mse = np.mean((y - y_pred) ** 2)
    residuals = y - y_pred
    
    # Calculate standard errors
    X_with_intercept = np.column_stack([np.ones(n), X])
    try:
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se = np.sqrt(np.diag(cov_matrix))
    except:
        # Fallback if matrix is singular
        se = np.full(p + 1, np.nan)
    
    # Calculate t-statistics and p-values
    coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))
    
    # Calculate AIC and BIC
    aic = n * np.log(mse) + 2 * (p + 1)
    bic = n * np.log(mse) + (p + 1) * np.log(n)
    
    return {
        'model': model,
        'r2': r2,
        'r2_adj': 1 - (1 - r2) * (n - 1) / (n - p - 1),
        'n': n,
        'coefs': coefs,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'aic': aic,
        'bic': bic,
        'feature_names': ['Intercept'] + feature_names
    }


def print_model_results(results, model_name):
    """Print model results in readable format."""
    print(f"\n{model_name}")
    print("="*70)
    print(f"{'Variable':<25} {'Coefficient':<15} {'Std Error':<12} {'t-stat':<10} {'p-value':<10}")
    print("-"*70)
    
    for i, name in enumerate(results['feature_names']):
        coef = results['coefs'][i]
        se = results['se'][i]
        t = results['t_stats'][i]
        p = results['p_values'][i]
        
        sig = ""
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        
        print(f"{name:<25} {coef:>10.4f}     {se:>8.4f}    {t:>8.4f}    {p:>8.4f} {sig}")
    
    print("-"*70)
    print(f"R-squared: {results['r2']:.4f}")
    print(f"Adj R-squared: {results['r2_adj']:.4f}")
    print(f"AIC: {results['aic']:.2f}")
    print(f"BIC: {results['bic']:.2f}")
    print(f"N: {results['n']}")


def run_model_1_basic(df: pd.DataFrame):
    """Model 1: Basic model - within_cluster only"""
    X = df[['within_cluster']].values
    y = df['offer'].values
    
    results = fit_linear_model(X, y, ['within_cluster'])
    print_model_results(results, "MODEL 1: Basic Model (offer ~ within_cluster)")
    
    return results


def run_model_2_similarity(df: pd.DataFrame):
    """Model 2: Control for job similarity"""
    if df['job_similarity'].isna().all():
        print("\n[SKIP] Model 2: No job similarity data available")
        return None
    
    df_clean = df[df['job_similarity'].notna()].copy()
    if len(df_clean) == 0:
        print("\n[SKIP] Model 2: No data with similarity scores")
        return None
    
    X = df_clean[['within_cluster', 'job_similarity']].values
    y = df_clean['offer'].values
    
    results = fit_linear_model(X, y, ['within_cluster', 'job_similarity'])
    print_model_results(results, "MODEL 2: Control for Job Similarity (offer ~ within_cluster + job_similarity)")
    
    return results


def run_model_3_interaction(df: pd.DataFrame):
    """Model 3: Interaction between within_cluster and similarity"""
    if df['job_similarity'].isna().all():
        print("\n[SKIP] Model 3: No job similarity data available")
        return None
    
    df_clean = df[df['job_similarity'].notna()].copy()
    if len(df_clean) == 0:
        print("\n[SKIP] Model 3: No data with similarity scores")
        return None
    
    df_clean['within_x_similarity'] = df_clean['within_cluster'] * df_clean['job_similarity']
    
    X = df_clean[['within_cluster', 'job_similarity', 'within_x_similarity']].values
    y = df_clean['offer'].values
    
    results = fit_linear_model(X, y, ['within_cluster', 'job_similarity', 'within_cluster:job_similarity'])
    print_model_results(results, "MODEL 3: Interaction Model (offer ~ within_cluster + job_similarity + within_cluster:job_similarity)")
    
    return results


def run_model_4_cluster_fixed_effects(df: pd.DataFrame):
    """Model 4: Fixed effects for proposer and responder clusters"""
    cluster_vars = [f'proposer_{k}' for k in CLUSTERS.keys() if k != 'TECH']
    cluster_vars += [f'responder_{k}' for k in CLUSTERS.keys() if k != 'TECH']
    
    X = df[['within_cluster'] + cluster_vars].values
    y = df['offer'].values
    
    feature_names = ['within_cluster'] + cluster_vars
    results = fit_linear_model(X, y, feature_names)
    print_model_results(results, "MODEL 4: Cluster Fixed Effects (offer ~ within_cluster + proposer_cluster + responder_cluster)")
    
    return results


def run_model_5_demographics(df: pd.DataFrame):
    """Model 5: Control for demographics (gender, age)"""
    has_gender = 'proposer_gender' in df.columns and 'responder_gender' in df.columns
    has_age = 'proposer_age' in df.columns and 'responder_age' in df.columns
    
    if not (has_gender or has_age):
        print("\n[SKIP] Model 5: No demographic data available")
        return None
    
    features = ['within_cluster']
    
    if has_gender:
        df['same_gender'] = (df['proposer_gender'] == df['responder_gender']).astype(int)
        # Create gender dummies (use proposer gender, Male as reference)
        if df['proposer_gender'].dtype == 'object':
            df['proposer_female'] = (df['proposer_gender'].str.upper().str[0] == 'F').astype(int)
            features.extend(['same_gender', 'proposer_female'])
        else:
            features.append('same_gender')
    
    if has_age:
        df['age_diff'] = abs(df['proposer_age'] - df['responder_age'])
        df['proposer_age_centered'] = df['proposer_age'] - df['proposer_age'].mean()
        features.extend(['age_diff', 'proposer_age_centered'])
    
    X = df[features].values
    y = df['offer'].values
    
    results = fit_linear_model(X, y, features)
    print_model_results(results, "MODEL 5: Control for Demographics")
    
    return results


def run_model_6_gender_dummies(df: pd.DataFrame):
    """Model 6: Gender dummy variables (from_men, to_men)"""
    has_gender = 'proposer_gender' in df.columns and 'responder_gender' in df.columns

    if not has_gender:
        print("\n[SKIP] Model 6: No gender data available")
        return None

    df_clean = df.copy()

    # Create gender dummies: 1 if male, 0 otherwise
    df_clean['from_men'] = (df_clean['proposer_gender'].str.upper().str[0] == 'M').astype(int)
    df_clean['to_men'] = (df_clean['responder_gender'].str.upper().str[0] == 'M').astype(int)

    X = df_clean[['from_men', 'to_men']].values
    y = df_clean['offer'].values

    results = fit_linear_model(X, y, ['from_men', 'to_men'])
    print_model_results(results, "MODEL 6: Gender Dummies (offer ~ from_men + to_men)")

    return results


def run_model_7_full_model(df: pd.DataFrame):
    """Model 7: Full model with all controls"""
    features = ['within_cluster']
    
    # Add similarity if available
    if 'job_similarity' in df.columns and not df['job_similarity'].isna().all():
        df_clean = df[df['job_similarity'].notna()].copy()
        if len(df_clean) > 0:
            df_clean['within_x_similarity'] = df_clean['within_cluster'] * df_clean['job_similarity']
            features.extend(['job_similarity', 'within_x_similarity'])
            df = df_clean
    
    # Add demographics if available
    if 'proposer_gender' in df.columns:
        df['same_gender'] = (df['proposer_gender'] == df['responder_gender']).astype(int)
        if df['proposer_gender'].dtype == 'object':
            df['proposer_female'] = (df['proposer_gender'].str.upper().str[0] == 'F').astype(int)
            features.extend(['same_gender', 'proposer_female'])
        else:
            features.append('same_gender')
    
    if 'proposer_age' in df.columns:
        df['age_diff'] = abs(df['proposer_age'] - df['responder_age'])
        df['proposer_age_centered'] = df['proposer_age'] - df['proposer_age'].mean()
        features.extend(['age_diff', 'proposer_age_centered'])
    
    # Add cluster fixed effects (excluding TECH as reference)
    cluster_vars = [f'proposer_{k}' for k in CLUSTERS.keys() if k != 'TECH']
    cluster_vars += [f'responder_{k}' for k in CLUSTERS.keys() if k != 'TECH']
    features.extend(cluster_vars)
    
    # Filter to complete cases
    available_features = [f for f in features if f in df.columns]
    df_clean = df[available_features + ['offer']].dropna()
    
    if len(df_clean) < len(df) * 0.5:
        print(f"\n[WARNING] Dropped {len(df) - len(df_clean)} cases due to missing data")
    
    if len(df_clean) < 10:
        print("\n[SKIP] Model 7: Too few complete cases")
        return None
    
    X = df_clean[available_features].values
    y = df_clean['offer'].values
    
    results = fit_linear_model(X, y, available_features)
    print_model_results(results, "MODEL 7: Full Model (All Controls)")
    
    return results


def compare_models(models: dict):
    """Compare model fit statistics"""
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison_data = []
    for name, results in models.items():
        if results is not None:
            within_idx = None
            for i, feat in enumerate(results['feature_names']):
                if 'within_cluster' in feat and ':' not in feat:
                    within_idx = i
                    break
            
            if within_idx is not None:
                comparison_data.append({
                    'Model': name,
                    'R-squared': results['r2'],
                    'Adj R-squared': results['r2_adj'],
                    'AIC': results['aic'],
                    'BIC': results['bic'],
                    'N': results['n'],
                    'within_cluster_coef': results['coefs'][within_idx],
                    'within_cluster_pval': results['p_values'][within_idx]
                })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print("\nModel Fit Statistics:")
        print(comp_df.to_string(index=False))
        
        print("\n\nWithin-Cluster Coefficient Across Models:")
        print(f"{'Model':<35} {'Coefficient':<15} {'P-value':<15} {'Significant'}")
        print("-" * 80)
        for _, row in comp_df.iterrows():
            sig = "***" if row['within_cluster_pval'] < 0.001 else "**" if row['within_cluster_pval'] < 0.01 else "*" if row['within_cluster_pval'] < 0.05 else ""
            print(f"{row['Model']:<35} ${row['within_cluster_coef']:>8.2f}      {row['within_cluster_pval']:>8.4f}      {sig}")


def main():
    parser = argparse.ArgumentParser(
        description='Run linear models for cluster generosity analysis'
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
        help='Path to clustered jobs JSON file'
    )
    parser.add_argument(
        '--similarities',
        type=str,
        default='cluster_similarities.json',
        help='Path to job similarities JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("LINEAR MODELS FOR CLUSTER GENEROSITY ANALYSIS")
    print("="*70)
    print(f"\nData file: {args.csv_file}")
    print(f"Cluster definitions: {args.clustered_jobs}")
    print(f"Similarities file: {args.similarities}")
    
    # Load data
    job_to_cluster = load_clustered_jobs(args.clustered_jobs)
    
    similarities = None
    try:
        similarities = load_job_similarities(args.similarities)
        print(f"\n[OK] Loaded job similarities")
    except Exception as e:
        print(f"\n[WARNING] Could not load job similarities: {e}")
    
    df = prepare_data(args.csv_file, job_to_cluster, similarities)
    print(f"\n[OK] Prepared data: {len(df)} observations")
    
    # Run models
    models = {}
    
    models['Model 1: Basic'] = run_model_1_basic(df)
    models['Model 2: + Similarity'] = run_model_2_similarity(df)
    models['Model 3: + Interaction'] = run_model_3_interaction(df)
    models['Model 4: + Cluster FE'] = run_model_4_cluster_fixed_effects(df)
    models['Model 5: + Demographics'] = run_model_5_demographics(df)
    models['Model 6: Gender Dummies'] = run_model_6_gender_dummies(df)
    models['Model 7: Full Model'] = run_model_7_full_model(df)
    
    # Compare models
    compare_models(models)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
