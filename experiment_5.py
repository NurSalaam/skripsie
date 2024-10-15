import cudf.pandas
cudf.pandas.install()

import sys
sys.path.insert(0, '/teamspace/studios/this_studio')
import pyscamp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import json
import re

from lof import LocalOutlierFactor
from matrix_profile import MatrixProfile
from isolation_forest import IsolationForest
from kmeans import KMeans
from ensemble_detector import EnsembleDetector
from dataloader import DataLoader
from benchmarker import benchmark

# Constants
UCR_PATH = 'ucrdata'
CACHED_SCORES_DIR = 'final_scores'
RESULTS_DIR = 'final_results'
ENSEMBLE_RESULTS_DIR = 'final_ensembles/oracle_results'
ENSEMBLE_SCORES_DIR = 'final_ensembles/oracle_scores'

# Ensemble methods
ENSEMBLE_METHODS = [
    ('simple_average', {}),
    ('maximum_score', {}),
    ('wv_ols', {}),
    ('wv_ols_r2', {'emphasize_diversity': False}),
    ('hard_voting', {'spread_window': 100, 'gaussian': False}),
    ('hard_voting', {'spread_window': 100, 'gaussian': True})
]

def instantiate_models(model_names):
    models = []
    
    for name in model_names:
        if match := re.match(r'lof_w(\d+)_n(\d+)', name):
            window_size, neighbors = map(int, match.groups())
            models.append(LocalOutlierFactor(windowSize=window_size, neighbors=neighbors, gpu=True))
        
        elif match := re.match(r'if_w(\d+)_n(\d+)', name):
            window_size, n_estimators = map(int, match.groups())
            models.append(IsolationForest(windowSize=window_size, n_estimators=n_estimators))
        
        elif match := re.match(r'km_w(\d+)_k(\d+)', name):
            window_size, n_clusters = map(int, match.groups())
            models.append(KMeans(windowSize=window_size, n_clusters=n_clusters))
        
        elif match := re.match(r'mp_w(\d+)', name):
            window_size = int(match.group(1))
            models.append(MatrixProfile(windowSize=window_size))
        
        else:
            raise ValueError(f"Unknown or improperly formatted model name: {name}")
    
    return models

def run_experiment():
    results_file = 'final_experiments/experiment_e5_results.csv'
    configs_file = 'final_experiments/experiment_e5_configurations.json'

    # Load best models
    best_models_df = pd.read_csv('final_visualisations/experiment_1/best_models_by_anomaly_type.csv')
    best_model_names = best_models_df['Best Model'].tolist()
    
    # Instantiate best models
    base_learners = instantiate_models(best_model_names)
    
    # Create or clear the results file with headers
    pd.DataFrame(columns=['ensemble_method', 'method_params', 'ucr_score', 'computational_time']).to_csv(results_file, index=False)

    # Clear the configurations file
    with open(configs_file, 'w') as f:
        json.dump([], f)

    results = []
    configs = []
    
    for method, method_params in tqdm(ENSEMBLE_METHODS, desc="Running ensemble methods"):
        ensemble = EnsembleDetector(
            base_learners=base_learners,
            method=method,
            method_params=method_params,
            scores_dir=CACHED_SCORES_DIR
        )
        
        ensemble_name = ensemble.toString()
        save_results_file = os.path.join(ENSEMBLE_RESULTS_DIR, f"{ensemble_name}_e5.csv")
        save_scores_dir = os.path.join(ENSEMBLE_SCORES_DIR, f"{ensemble_name}_e5")
        
        print(f"Processing: {ensemble_name}")
        benchmark(ensemble, UCR_PATH, save_results_file, save_scores_dir)
        
        benchmark_results = pd.read_csv(save_results_file, nrows=1)
        
        result = {
            'ensemble_method': method,
            'method_params': json.dumps(method_params),
            'ucr_score': benchmark_results['accuracy'].values[0],
            'computational_time': benchmark_results['total_time'].values[0]
        }
        results.append(result)
        
        configuration = {
            'ensemble_method': method,
            'method_params': method_params,
            'base_learner_params': best_model_names
        }
        configs.append(configuration)
    
    # Save results to CSV file
    pd.DataFrame(results).to_csv(results_file, mode='a', header=False, index=False)
    
    # Save configurations to JSON file
    with open(configs_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print("Experiment E5 completed. Results and configurations saved.")

def analyze_results(results_df):
    summary = results_df.groupby(['ensemble_method', 'method_params']).agg({
        'ucr_score': ['mean', 'std', 'max'],
        'computational_time': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['ensemble_method', 'method_params', 'mean_score', 'std_score', 'max_score', 'mean_time', 'std_time']
    summary['cv_score'] = summary['std_score'] / summary['mean_score']
    
    return summary

# Run the experiment
if not os.path.exists(ENSEMBLE_RESULTS_DIR):
    os.makedirs(ENSEMBLE_RESULTS_DIR)
if not os.path.exists(ENSEMBLE_SCORES_DIR):
    os.makedirs(ENSEMBLE_SCORES_DIR)

run_experiment()

# Analyze the results
results_df = pd.read_csv('final_experiments/experiment_e5_results.csv')
summary = analyze_results(results_df)
summary.to_csv('final_experiments/experiment_e5_summary.csv', index=False)

print("Analysis completed. Summary saved to 'final_experiments/experiment_e5_summary.csv'.")

# Load and print summary
summary = pd.read_csv('final_experiments/experiment_e5_summary.csv')
print(summary)