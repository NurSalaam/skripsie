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
from kmeans import KMeans
from ensemble_detector import EnsembleDetector
from dataloader import DataLoader
from benchmarker import benchmark

# Constants
UCR_PATH = 'ucrdata'
CACHED_SCORES_DIR = 'final_scores'
RESULTS_DIR = 'final_results'
ENSEMBLE_RESULTS_DIR = 'final_ensembles/e4_results'
ENSEMBLE_SCORES_DIR = 'final_ensembles/e4_scores'

# Ensemble methods
ENSEMBLE_METHODS = [
    ('simple_average', {}),
    ('hard_voting', {'spread_window': 100, 'gaussian': False}),
    ('hard_voting', {'spread_window': 100, 'gaussian': True})
]

# Base learner combinations
BASE_LEARNER_COMBINATIONS = [
    ('LOF', 'MP'),
    ('LOF', 'KMeans'),
    ('KMeans', 'MP')
]

def instantiate_models(model_type):
    models = []
    
    if model_type == 'LOF':
        for window_size in [25, 50, 100, 150, 200]:
            for neighbors in [10, 20, 50, 100]:
                models.append(LocalOutlierFactor(windowSize=window_size, neighbors=neighbors, gpu=True))
    elif model_type == 'MP':
        for window_size in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]:
            models.append(MatrixProfile(windowSize=window_size))
    elif model_type == 'KMeans':
        for window_size in [50, 100, 200, 500]:
            for n_clusters in [10, 20, 50, 100, 200]:
                models.append(KMeans(windowSize=window_size, n_clusters=n_clusters))
    
    return models

def run_experiment():
    results_file = 'final_experiments/experiment_e4_results.csv'
    configs_file = 'final_experiments/experiment_e4_configurations.json'

    # Create or clear the results file with headers
    pd.DataFrame(columns=['combination', 'ensemble_method', 'method_params', 'ucr_score', 'computational_time']).to_csv(results_file, index=False)

    # Clear the configurations file
    with open(configs_file, 'w') as f:
        json.dump([], f)

    results = []
    configs = []
    
    for combo in BASE_LEARNER_COMBINATIONS:
        base_learners = instantiate_models(combo[0]) + instantiate_models(combo[1])
        
        for method, method_params in tqdm(ENSEMBLE_METHODS, desc=f"Running ensemble methods for {combo}"):
            ensemble = EnsembleDetector(
                base_learners=base_learners,
                method=method,
                method_params=method_params,
                scores_dir=CACHED_SCORES_DIR
            )
            
            ensemble_name = f"{combo[0]}_{combo[1]}_{ensemble.toString()}"
            save_results_file = os.path.join(ENSEMBLE_RESULTS_DIR, f"{ensemble_name}_e4.csv")
            save_scores_dir = os.path.join(ENSEMBLE_SCORES_DIR, f"{ensemble_name}_e4")
            
            print(f"Processing: {ensemble_name}")
            benchmark(ensemble, UCR_PATH, save_results_file, save_scores_dir)
            
            benchmark_results = pd.read_csv(save_results_file, nrows=1)
            
            result = {
                'combination': f"{combo[0]}-{combo[1]}",
                'ensemble_method': method,
                'method_params': json.dumps(method_params),
                'ucr_score': benchmark_results['accuracy'].values[0],
                'computational_time': benchmark_results['total_time'].values[0]
            }
            results.append(result)
            
            configuration = {
                'combination': f"{combo[0]}-{combo[1]}",
                'ensemble_method': method,
                'method_params': method_params,
                'base_learner_params': [str(bl) for bl in base_learners]
            }
            configs.append(configuration)
    
    # Save results to CSV file
    pd.DataFrame(results).to_csv(results_file, mode='a', header=False, index=False)
    
    # Save configurations to JSON file
    with open(configs_file, 'w') as f:
        json.dump(configs, f, indent=2)

    print("Experiment E4 completed. Results and configurations saved.")

def analyze_results(results_df):
    summary = results_df.groupby(['combination', 'ensemble_method', 'method_params']).agg({
        'ucr_score': ['mean', 'std', 'max'],
        'computational_time': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['combination', 'ensemble_method', 'method_params', 'mean_score', 'std_score', 'max_score', 'mean_time', 'std_time']
    summary['cv_score'] = summary['std_score'] / summary['mean_score']
    
    return summary

# Run the experiment
if not os.path.exists(ENSEMBLE_RESULTS_DIR):
    os.makedirs(ENSEMBLE_RESULTS_DIR)
if not os.path.exists(ENSEMBLE_SCORES_DIR):
    os.makedirs(ENSEMBLE_SCORES_DIR)

run_experiment()

# Analyze the results
results_df = pd.read_csv('final_experiments/experiment_e4_results.csv')
summary = analyze_results(results_df)
summary.to_csv('final_experiments/experiment_e4_summary.csv', index=False)

print("Analysis completed. Summary saved to 'final_experiments/experiment_e4_summary.csv'.")

# Load and print summary
summary = pd.read_csv('final_experiments/experiment_e4_summary.csv')
print(summary)