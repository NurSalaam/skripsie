import cudf.pandas
cudf.pandas.install()

import pyscamp

import numpy as np
import pandas as pd
from tqdm import tqdm

from lof import LocalOutlierFactor
from matrix_profile import MatrixProfile
from isolation_forest import IsolationForest
from kmeans import KMeans

from dataloader import DataLoader
from benchmarker import benchmark, process_precomputed_scores
import os
import itertools

# Define base learners and their parameter ranges
base_learners = {
    'LOF': {
        'class': LocalOutlierFactor,
        'params': {
            'windowSize': [25, 50, 100, 150, 200, 250], #[25, 250], #
            'neighbors': [10, 20, 50, 100], #[50, 100], #
            'gpu': [True]
        }
    },
    'IF': {
        'class': IsolationForest,
        'params': {
            'windowSize': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        }
    },
    'KMeans': {
        'class': KMeans,
        'params': {
            'windowSize': [50, 100, 200, 500],
            'n_clusters': [10, 20, 50, 100, 200],
        }
    },
    'MP': {
        'class': MatrixProfile,
        'params': {
            'windowSize': [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        }
    }
}

def run_experiment(base_learner_name, base_learner_class, params, ucr_path, results_path, scores_path):
    """Run experiment for a single base learner with given parameters."""
    learner = base_learner_class(**params)
    benchmark_results_file = os.path.join(results_path, f"{learner.toString()}.csv")
    benchmark_scores_dir = os.path.join(scores_path, f"{learner.toString()}")
    
    # run benchmark
    benchmark(learner, ucr_path, benchmark_results_file, benchmark_scores_dir)

    # Load results
    summary = pd.read_csv(benchmark_results_file, nrows=1)
    
    
    return {
        'params': params,
        'ucr_score': summary['accuracy'].values[0],
        'computational_time': summary['total_time'].values[0],
    }

def generate_param_combinations(param_dict):
    """Generate all combinations of parameters."""
    keys = list(param_dict.keys())
    values = list(param_dict.values())
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

# Main experiment loop
ucr_path = 'ucrdata'
results_path = 'final_results'
scores_path = 'final_scores'

all_results = []

for base_learner_name, base_learner_info in base_learners.items():
    print(f"Running experiments for {base_learner_name}")
    base_learner_class = base_learner_info['class']
    param_combinations = list(generate_param_combinations(base_learner_info['params']))
    
    for params in tqdm(param_combinations, desc=f"{base_learner_name} configurations"):
        result = run_experiment(base_learner_name, base_learner_class, params, ucr_path, results_path, scores_path)
        result['base_learner'] = base_learner_name
        all_results.append(result)

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Analysis
# for base_learner_name in base_learners.keys():
#     learner_results = results_df[results_df['base_learner'] == base_learner_name]
    
#     print(f"\nAnalysis for {base_learner_name}:")
#     for metric in ['ucr_score', 'computational_time']:
#         mean = learner_results[metric].mean()
#         std = learner_results[metric].std()
#         cv = std / mean
#         print(f"{metric}:")
#         print(f"  Mean: {mean:.4f}")
#         print(f"  Std Dev: {std:.4f}")
#         print(f"  Coeff of Variation: {cv:.4f}")
    
#     best_config = learner_results.loc[learner_results['ucr_score'].idxmax()]
#     print(f"\nBest configuration for {base_learner_name}:")
#     print(best_config['params'])
#     print(f"UCR Score: {best_config['ucr_score']:.4f}")

# Save full results
experiments_path = os.path.join('final_experiments', 'experiment_e1_results.csv')
results_df.to_csv(experiments_path, index=False)
print(f"\nFull results saved to {experiments_path}")