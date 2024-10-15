import cudf.pandas
cudf.pandas.install()
# import cProfile
# import pstats
import pyscamp
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
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
CACHED_SCORES_DIR = 'final_scores'
RESULTS_DIR = 'final_results'
ENSEMBLE_RESULTS_DIR = 'final_ensembles/results'
ENSEMBLE_SCORES_DIR = 'final_ensembles/scores'
# N_REPETITIONS = 30
N_REPETITIONS = 10
ENSEMBLE_SIZE = 16

# Base learner configurations
BASE_LEARNERS = {
    # 'LOF': {
    #     'class': LocalOutlierFactor,
    #     'params': [
    #         {'windowSize': ws, 'neighbors': n, 'gpu': True}
    #         for ws in [25, 50, 100, 150, 200]
    #         for n in [10, 20, 50, 100]
    #     ]
    # }
    'IF': {
        'class': IsolationForest,
        'params': [
            {'windowSize': ws}
            for ws in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
        ]
    }#,
    # 'KMeans': {
    #     'class': KMeans,
    #     'params': [
    #         {'windowSize': ws, 'n_clusters': nc}
    #         for ws in [50, 100, 200, 500]
    #         for nc in [10, 20, 50, 100, 200]
    #     ]
    # },
    # 'MP': {
    #     'class': MatrixProfile,
    #     'params': [
    #         {'windowSize': ws}
    #         for ws in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
    #     ]
    # }
}

# Ensemble methods
ENSEMBLE_METHODS = [
    ('simple_average', {}),
    ('maximum_score', {}),
    ('wv_ols', {}),
    ('wv_ols_r2', {'emphasize_diversity': False}),
    # ('wv_ols_r2', {'emphasize_diversity': True}),
    ('hard_voting', {'spread_window': 100, 'gaussian': False}),
    ('hard_voting', {'spread_window': 100, 'gaussian': True})
]

def create_homogeneous_ensemble(base_learner_type, n=ENSEMBLE_SIZE):
    base_learner_class = BASE_LEARNERS[base_learner_type]['class']
    params_list = BASE_LEARNERS[base_learner_type]['params']
    selected_params = random.sample(params_list, n)
    return [base_learner_class(**params) for params in selected_params], selected_params

# def run_experiment():
#     results = []
#     configurations = []
    
#     for base_learner_type in BASE_LEARNERS.keys():
#         for rep in tqdm(range(N_REPETITIONS), desc=f"Running {base_learner_type} ensembles"):
#             base_learners, selected_params = create_homogeneous_ensemble(base_learner_type)
            
#             for method, method_params in ENSEMBLE_METHODS:
#                 ensemble = EnsembleDetector(
#                     base_learners=base_learners,
#                     method=method,
#                     method_params=method_params,
#                     scores_dir=CACHED_SCORES_DIR
#                 )
                
#                 ensemble_name = ensemble.toString()
#                 save_results_file = os.path.join(ENSEMBLE_RESULTS_DIR, f"{ensemble_name}_rep{rep}.csv")
#                 save_scores_dir = os.path.join(ENSEMBLE_SCORES_DIR, f"{ensemble_name}_rep{rep}")
                
#                 benchmark(ensemble, UCR_PATH, save_results_file, save_scores_dir)
                
#                 benchmark_results = pd.read_csv(save_results_file, nrows=1)
                
#                 result = {
#                     'base_learner_type': base_learner_type,
#                     'ensemble_method': method,
#                     'method_params': json.dumps(method_params),  # Convert dict to string for storage in DataFrame
#                     'repetition': rep,
#                     'ucr_score': benchmark_results['accuracy'].values[0],
#                     'computational_time': benchmark_results['total_time'].values[0]
#                 }
#                 results.append(result)
                
#                 configuration = {
#                     'base_learner_type': base_learner_type,
#                     'ensemble_method': method,
#                     'method_params': method_params,
#                     'repetition': rep,
#                     'base_learner_params': selected_params
#                 }
#                 configurations.append(configuration)
    
#     return pd.DataFrame(results), 
# @profile
def run_experiment():
    results_file = 'final_experiments/experiment_e2_results.csv'
    configs_file = 'final_experiments/experiment_e2_configurations.json'

    # Create or clear the results file with headers
    # pd.DataFrame(columns=['base_learner_type', 'ensemble_method', 'method_params', 'repetition', 'ucr_score', 'computational_time']).to_csv(results_file, index=False)

    # # Clear the configurations file
    # with open(configs_file, 'w') as f:
    #     json.dump([], f)

    for base_learner_type in BASE_LEARNERS.keys():
        for rep in tqdm(range(N_REPETITIONS), desc=f"Running {base_learner_type} ensembles"):
            base_learners, selected_params = create_homogeneous_ensemble(base_learner_type)
            
            rep_results = []
            rep_configs = []
            
            for method, method_params in ENSEMBLE_METHODS:
                ensemble = EnsembleDetector(
                    base_learners=base_learners,
                    method=method,
                    method_params=method_params,
                    scores_dir=CACHED_SCORES_DIR
                )
                
                ensemble_name = ensemble.toString()
                save_results_file = os.path.join(ENSEMBLE_RESULTS_DIR, f"{ensemble_name}_rep{rep}.csv")
                save_scores_dir = os.path.join(ENSEMBLE_SCORES_DIR, f"{ensemble_name}_rep{rep}")
                
                benchmark(ensemble, UCR_PATH, save_results_file, save_scores_dir)
                
                benchmark_results = pd.read_csv(save_results_file, nrows=1)
                
                result = {
                    'base_learner_type': base_learner_type,
                    'ensemble_method': method,
                    'method_params': json.dumps(method_params),
                    'repetition': rep,
                    'ucr_score': benchmark_results['accuracy'].values[0],
                    'computational_time': benchmark_results['total_time'].values[0]
                }
                rep_results.append(result)
                
                configuration = {
                    'base_learner_type': base_learner_type,
                    'ensemble_method': method,
                    'method_params': method_params,
                    'repetition': rep,
                    'base_learner_params': selected_params
                }
                rep_configs.append(configuration)
            
            # Append results for this repetition to the CSV file
            pd.DataFrame(rep_results).to_csv(results_file, mode='a', header=False, index=False)
            
            # Append configurations for this repetition to the JSON file
            with open(configs_file, 'r+') as f:
                configs = json.load(f)
                configs.extend(rep_configs)
                f.seek(0)
                json.dump(configs, f, indent=2)
                f.truncate()

    print("Experiment E2 completed. Results and configurations saved incrementally.")

def analyze_results(results_df):
    summary = results_df.groupby(['base_learner_type', 'ensemble_method', 'method_params']).agg({
        'ucr_score': ['mean', 'std', 'max'],
        'computational_time': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['base_learner_type', 'ensemble_method', 'method_params', 'mean_score', 'std_score', 'max_score', 'mean_time', 'std_time']
    summary['cv_score'] = summary['std_score'] / summary['mean_score']
    
    return summary

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()

# Run the experiment
if not os.path.exists(ENSEMBLE_RESULTS_DIR):
    os.makedirs(ENSEMBLE_RESULTS_DIR)
if not os.path.exists(ENSEMBLE_SCORES_DIR):
    os.makedirs(ENSEMBLE_SCORES_DIR)

run_experiment()

# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('cumulative')
# stats.print_stats(50) 
# results_df.to_csv('final_experiments/experiment_e2_results.csv', index=False)


# # Save configurations
# # with open('final_experiments/experiment_e2_configurations.json', 'w') as f:
# with open('final_experiments/experiment_e2_configurations.json', 'w') as f:
#     json.dump(configurations, f, indent=2)

# # Analyze the results
# summary = analyze_results(results_df)
# # summary.to_csv('final_experiments/experiment_e2_summary.csv', index=False)
# summary.to_csv('final_experiments/experiment_e2_summary.csv', index=False)


# # print("Experiment E2 completed. Results saved to 'final_experiments/experiment_e2_results.csv', 'final_experiments/experiment_e2_configurations.json', and 'final_experiments/experiment_e2_summary.csv'.")

# print("Experiment E2 completed. Results saved to 'final_experiments/experiment_e2_results.csv', 'final_experiments/experiment_e2_configurations.json', and 'final_experiments/experiment_e2_summary.csv'.")

