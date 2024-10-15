
import pyscamp
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
import json

from lof import LocalOutlierFactor
from matrix_profile import MatrixProfile
from isolation_forest import IsolationForest
from kmeans import KMeans
from ensemble_detector import EnsembleDetector
from dataloader import DataLoader
from benchmarker import benchmark

# Constants
UCR_PATH = 'ucrdata'
RESULTS_PATH = 'results'
SCORES_PATH = 'scores'
ENSEMBLE_RESULTS_DIR = 'test/ensembles/results'
ENSEMBLE_SCORES_DIR = 'test/ensembles/scores'

# Define base learners and their parameter ranges
base_learner_configs = {
    'LOF': {
        'class': LocalOutlierFactor,
        'params': [
            {'windowSize': ws, 'neighbors': n, 'gpu': True}
            for ws in [25, 50, 100, 150, 200, 250]
            for n in [10, 20, 50, 100]
        ]
    },
    'IF': {
        'class': IsolationForest,
        'params': [
            {'windowSize': ws}
            for ws in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        ]
    },
    'KMeans': {
        'class': KMeans,
        'params': [
            {'windowSize': ws, 'n_clusters': nc}
            for ws in [50, 100, 200, 500]
            for nc in [10, 20, 50, 100, 200]
        ]
    },
    'MP': {
        'class': MatrixProfile,
        'params': [
            {'windowSize': ws}
            for ws in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        ]
    }
}

# Ensemble methods
ENSEMBLE_METHODS = [
    ('simple_average', {}),
    # ('maximum_score', {}),
    ('wv_ols', {}),
    # ('wv_ols_r2', {'emphasize_diversity': False}),
    # ('wv_ols_r2', {'emphasize_diversity': True}),
    ('hard_voting', {'spread_window': 100, 'gaussian': False}),
    # ('hard_voting', {'spread_window': 100, 'gaussian': True})
]

def create_pairwise_ensemble(base_learner_type1, base_learner_type2):
    base_learner_instances = []
    selected_params = []
    
    for params in base_learner_configs[base_learner_type1]['params']:
        base_learner_instances.append(base_learner_configs[base_learner_type1]['class'](**params))
        selected_params.append((base_learner_type1, params))
    
    for params in base_learner_configs[base_learner_type2]['params']:
        base_learner_instances.append(base_learner_configs[base_learner_type2]['class'](**params))
        selected_params.append((base_learner_type2, params))
    
    return base_learner_instances, selected_params

def run_experiment():
    results = []
    configurations = []
    
    base_learner_pairs = list(itertools.combinations(base_learner_configs.keys(), 2))
    
    for pair in tqdm(base_learner_pairs, desc="Running pairwise ensembles"):
        base_learner_instances, selected_params = create_pairwise_ensemble(pair[0], pair[1])
        
        for method, method_params in ENSEMBLE_METHODS:
            ensemble = EnsembleDetector(
                base_learners=base_learner_instances,
                method=method,
                method_params=method_params,
                scores_dir=SCORES_PATH
            )
            
            ensemble_name = f"ensemble_{pair[0]}_{pair[1]}_{ensemble.toString()}"
            save_results_file = os.path.join(ENSEMBLE_RESULTS_DIR, f"{ensemble_name}.csv")
            save_scores_dir = os.path.join(ENSEMBLE_SCORES_DIR, ensemble_name)
            
            benchmark(ensemble, UCR_PATH, save_results_file, save_scores_dir)
            
            benchmark_results = pd.read_csv(save_results_file, nrows=1)
            
            result = {
                'base_learner_pair': f"{pair[0]}_{pair[1]}",
                'ensemble_method': method,
                'method_params': json.dumps(method_params),
                'ucr_score': benchmark_results['accuracy'].values[0],
                'computational_time': benchmark_results['total_time'].values[0]
            }
            results.append(result)
            
            configuration = {
                'base_learner_pair': f"{pair[0]}_{pair[1]}",
                'ensemble_method': method,
                'method_params': method_params,
                'base_learner_params': selected_params
            }
            configurations.append(configuration)
    
    return pd.DataFrame(results), configurations

# Ensure directories exist
os.makedirs(ENSEMBLE_RESULTS_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_SCORES_DIR, exist_ok=True)

# Run the experiment
results_df, configurations = run_experiment()

# Save results
results_df.to_csv('test/experiments/experiment_e1b_results.csv', index=False)

# Save configurations
with open('test/experiments/experiment_e1b_configurations.json', 'w') as f:
    json.dump(configurations, f, indent=2)

print("Experiment E1b completed. Results saved to 'test/experiments/experiment_e1b_results.csv' and 'test/experiments/experiment_e1b_configurations.json'.")