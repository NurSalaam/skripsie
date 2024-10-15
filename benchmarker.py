import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from dataloader import DataLoader

def benchmark(algorithm, path_to_benchmark, path_to_save_results_file, path_to_save_scores, save_scores=True):
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(path_to_save_results_file), exist_ok=True)
    
    # Create scores directory if it doesn't exist
    scores_dir = os.path.join(path_to_save_scores)
    os.makedirs(scores_dir, exist_ok=True)
    
    results = []
    total_time = 0
    n_correct = 0
    n_incorrect = 0
    n_failed = 0
    
    # Load all time series from the benchmark directory
    ts_files = [f for f in os.listdir(path_to_benchmark) if f.endswith('.csv')]
    
    for ts_file in tqdm(ts_files, desc="Benchmarking"):
        file_path = os.path.join(path_to_benchmark, ts_file)
        
        try:
            # Load time series
            ts = DataLoader().load_file(file_path)
            
            # Fit the algorithm
            # algorithm.fit(ts)
            
            # Predict
            ts, processing_time = algorithm.predict(ts)
            
            total_time += processing_time
            
            # Get prediction
            scores = ts.testData["Score"]
            prediction = np.argmax(scores)
            
            # Check if prediction is correct
            correct = any(anomaly.start <= prediction <= anomaly.end for anomaly in ts.anomalies)
            
            if correct:
                n_correct += 1
                status = "correct"
            else:
                n_incorrect += 1
                status = "incorrect"
            
            # Save scores
            if save_scores:
                scores_file = os.path.join(scores_dir, f"{ts.name}_scores.csv")
                pd.DataFrame({"Scores": scores}).to_csv(scores_file, index=False)
            
            # Append result
            for anomaly in ts.anomalies:
                results.append({
                    "ts_name": ts.name,
                    "status": status,
                    "predicted_anomaly": prediction,
                    "anomaly_start": anomaly.start,
                    "anomaly_end": anomaly.end,
                    "processing_time": processing_time
                })
        
        except Exception as e:
            print(f"Failed to process {ts_file}: {str(e)}")
            n_failed += 1
            results.append({
                "ts_name": ts_file,
                "status": "failed",
                "predicted_anomaly": None,
                "anomaly_start": None,
                "anomaly_end": None,
                "processing_time": None
            })
    
    # Calculate accuracy
    total_processed = n_correct + n_incorrect
    accuracy = n_correct / total_processed if total_processed > 0 else 0
    
    # Save results
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame({
        "algorithm_name": [algorithm.toString()],
        "total_time": [total_time],
        "accuracy": [accuracy],
        "n_correct": [n_correct],
        "n_incorrect": [n_incorrect],
        "n_failed": [n_failed]
    })
    
    with open(path_to_save_results_file, 'w') as f:
        summary_df.to_csv(f, index=False)
        f.write("###\n")
        results_df.to_csv(f, index=False)
    
    # Print summary
    print(f"Benchmark complete for {algorithm.toString()}:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct predictions: {n_correct}")
    print(f"Incorrect predictions: {n_incorrect}")
    print(f"Failed predictions: {n_failed}")
    print(f"Results saved to: {path_to_save_results_file}")
    if save_scores:
        print(f"Scores saved to: {scores_dir}")

# Example usage:
# benchmark(LocalOutlierFactor(gpu=True), "path/to/benchmark", "path/to/results.csv", "path/to/scores")



def process_precomputed_scores(algorithm_name, path_to_benchmark, path_to_scores, path_to_save_results_file):
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(path_to_save_results_file), exist_ok=True)
    
    results = []
    total_time = 0
    n_correct = 0
    n_incorrect = 0
    n_failed = 0
    
    # Load all time series from the benchmark directory
    ts_files = [f for f in os.listdir(path_to_benchmark) if f.endswith('.csv')]
    
    for ts_file in tqdm(ts_files, desc="Processing"):
        file_path = os.path.join(path_to_benchmark, ts_file)
        scores_file = os.path.join(path_to_scores, f"{ts_file.split('.')[0]}_scores.csv")
        
        try:
            # Load time series
            ts = DataLoader().load_file(file_path)
            
            # Load pre-computed scores
            scores_df = pd.read_csv(scores_file)
            try:
                scores = scores_df["Score"].values
            except Exception as e:
                scores = scores_df['Scores'].values
            
            # Get prediction
            prediction = np.argmax(scores)
            
            # Check if prediction is correct
            correct = any(anomaly.start <= prediction <= anomaly.end for anomaly in ts.anomalies)
            
            if correct:
                n_correct += 1
                status = "correct"
            else:
                n_incorrect += 1
                status = "incorrect"
            
            # Append result
            for anomaly in ts.anomalies:
                results.append({
                    "ts_name": ts.name,
                    "status": status,
                    "predicted_anomaly": prediction,
                    "anomaly_start": anomaly.start,
                    "anomaly_end": anomaly.end,
                    "processing_time": None  # We don't have this information for pre-computed scores
                })
        
        except Exception as e:
            print(f"Failed to process {ts_file}: {str(e)}")
            n_failed += 1
            results.append({
                "ts_name": ts_file,
                "status": "failed",
                "predicted_anomaly": None,
                "anomaly_start": None,
                "anomaly_end": None,
                "processing_time": None
            })
    
    # Calculate accuracy
    total_processed = n_correct + n_incorrect
    accuracy = n_correct / total_processed if total_processed > 0 else 0
    
    # Save results
    results_df = pd.DataFrame(results)
    summary_df = pd.DataFrame({
        "algorithm_name": [algorithm_name],
        "total_time": [None],  # We don't have this information for pre-computed scores
        "accuracy": [accuracy],
        "n_correct": [n_correct],
        "n_incorrect": [n_incorrect],
        "n_failed": [n_failed]
    })
    
    with open(path_to_save_results_file, 'w') as f:
        summary_df.to_csv(f, index=False)
        f.write("###\n")
        results_df.to_csv(f, index=False)
    
    # Print summary
    print(f"Processing complete for {algorithm_name}:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Correct predictions: {n_correct}")
    print(f"Incorrect predictions: {n_incorrect}")
    print(f"Failed predictions: {n_failed}")
    print(f"Results saved to: {path_to_save_results_file}")

# Example usage:
# process_precomputed_scores("LOF_w100_n20", "path/to/benchmark", "path/to/scores", "path/to/results.csv")