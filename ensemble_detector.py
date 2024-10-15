import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata, norm
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import List, Tuple, Dict
import time

from dataloader import TimeSeries, DataLoader

class EnsembleDetector:
    def __init__(self, base_learners: List, method: str = 'simple_average', method_params: Dict = None, scores_dir:str='final_scores', results_dir:str='final_results'):
        self.base_learners = base_learners
        self.method = method
        self.method_params = method_params or {}
        self.scores_dir = scores_dir
        self.results_dir = results_dir

        self.method_map = {
            'simple_average': self.simple_average,
            'maximum_score': self.maximum_score,
            'wv_ols': self.wv_ols,
            'wv_ols_r2': self.wv_ols_r2,
            'wv_knn': self.wv_knn,
            'wv_ols_r2_topk': self.wv_ols_r2_topk,
            'hard_voting': self.hard_voting
        }

    def fit(self, tsObject: TimeSeries):
        for learner in self.base_learners:
            learner.fit(tsObject)

    def predict(self, tsObject: TimeSeries) -> Tuple[TimeSeries, float]:
        if self.method not in self.method_map:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self.method_map[self.method](tsObject, **self.method_params)

    def _get_base_learner_scores(self, tsObject: TimeSeries) -> Tuple[np.ndarray, float]:
        scores = []
        max_time = 0

        for learner in self.base_learners:
            score_file = f"{self.scores_dir}/{learner.toString()}/{tsObject.name}_scores.csv"
            benchmark_results_file = os.path.join(self.results_dir, f"{learner.toString()}.csv")
            if os.path.exists(benchmark_results_file):
                df = pd.read_csv(benchmark_results_file, skiprows=3)
                processing_time = df[df['ts_name']==tsObject.name]['processing_time']
                if not processing_time.empty:
                    max_time = max(max_time, processing_time.iloc[0])  # Use the first value
                else:
                    print(f"Warning: No processing time found for {tsObject.name} in {learner.toString()}")

            if os.path.exists(score_file):
                # Use pandas to read the CSV file, skipping the header
                try:
                    learner_scores = pd.read_csv(score_file, header=0)['Score'].values
                except:
                    # THIS IS SOME TRASH!!! 
                     learner_scores = pd.read_csv(score_file, header=0)['Scores'].values
                   
            else:
                print(f"Learner {learner.toString()} does not exist. Running predictions...")
                # Uncomment the following lines if you want to generate scores when they don't exist
                # result, learner_time = learner.predict(tsObject)
                # learner_scores = result.testData["Score"].values
                # os.makedirs(os.path.dirname(score_file), exist_ok=True)
                # pd.DataFrame(learner_scores, columns=['Score']).to_csv(score_file, index=False)
                # max_time = max(max_time, learner_time)
                raise FileNotFoundError(f"Score file not found: {score_file}")

            scores.append(learner_scores)

        return np.column_stack(scores), max_time

    def simple_average(self, tsObject: TimeSeries) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        avg_scores = np.mean(scores, axis=1)
        tsObject.testData['Score'] = minmax_scale(avg_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)
    
    def maximum_score(self, tsObject: TimeSeries) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        max_scores = np.max(scores, axis=1)
        tsObject.testData['Score'] = minmax_scale(max_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def wv_knn(self, tsObject: TimeSeries, k: int = 3) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        n_samples, n_detectors = scores.shape
        weights = np.zeros(n_detectors)
        
        for d in range(n_detectors):
            X = np.delete(scores, d, axis=1)
            y = scores[:, d]
            
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X, y)
            y_pred = knn.predict(X)
            
            weights[d] = r2_score(y, y_pred)
        
        weights = weights / np.sum(weights)
        final_scores = np.max(weights * scores, axis=1)
        
        tsObject.testData['Score'] = minmax_scale(final_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def wv_ols(self, tsObject: TimeSeries) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        n_samples, n_detectors = scores.shape
        weights = np.zeros(n_detectors)
        
        for d in range(n_detectors):
            X = np.delete(scores, d, axis=1)
            y = scores[:, d]
            
            ols = LinearRegression()
            ols.fit(X, y)
            y_pred = ols.predict(X)
            
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            weights[d] = max(0, 1 - rmse)
        
        weights = weights / np.sum(weights)
        final_scores = np.max(weights * scores, axis=1)
        
        tsObject.testData['Score'] = minmax_scale(final_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def wv_ols_r2(self, tsObject: TimeSeries, emphasize_diversity: bool = False) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        n_samples, n_detectors = scores.shape
        weights = np.zeros(n_detectors)
        
        for d in range(n_detectors):
            X = np.delete(scores, d, axis=1)
            y = scores[:, d]
            
            ols = LinearRegression()
            ols.fit(X, y)
            y_pred = ols.predict(X)
            
            weights[d] = r2_score(y, y_pred)
        
        if emphasize_diversity:
            weights = 1 - weights
        
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
        
        final_scores = np.max(weights * scores, axis=1)
        
        tsObject.testData['Score'] = minmax_scale(final_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def wv_ols_r2_topk(self, tsObject: TimeSeries, k: int) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        n_samples, n_detectors = scores.shape
        r2_scores = np.zeros(n_detectors)
        
        for d in range(n_detectors):
            X = np.delete(scores, d, axis=1)
            y = scores[:, d]
            
            ols = LinearRegression()
            ols.fit(X, y)
            y_pred = ols.predict(X)
            
            r2_scores[d] = r2_score(y, y_pred)
        
        top_k_indices = np.argsort(r2_scores)[:k]
        final_scores = np.mean(scores[:, top_k_indices], axis=1)
        
        tsObject.testData['Score'] = minmax_scale(final_scores)
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def hard_voting(self, tsObject: TimeSeries, spread_window: int = 100, gaussian: bool = False) -> Tuple[TimeSeries, float]:
        scores, max_time = self._get_base_learner_scores(tsObject)
        start_time = time.time()
        anomaly_indices = np.argmax(scores, axis=0)
        length = len(tsObject.testData)
        
        if gaussian:
            x = np.arange(-spread_window // 2, spread_window // 2 + 1)
            gaussian_kernel = norm.pdf(x, 0, spread_window / 6)
            gaussian_kernel /= gaussian_kernel.sum()
            
            indices = np.arange(length)
            distances = np.abs(indices[np.newaxis, :] - anomaly_indices[:, np.newaxis])
            
            mask = distances <= spread_window // 2
            
            spread_mask = np.zeros((len(self.base_learners), length))
            spread_mask[mask] = gaussian_kernel[distances[mask]]
        else:
            spread_mask = np.zeros((len(self.base_learners), length))
            
            starts = np.maximum(anomaly_indices - spread_window // 2, 0)
            ends = np.minimum(anomaly_indices + spread_window // 2 + 1, length)
            
            for i, (start, end) in enumerate(zip(starts, ends)):
                spread_mask[i, start:end] = 1
        
        combined_votes = np.sum(spread_mask, axis=0)
        combined_votes = (combined_votes - combined_votes.min()) / (combined_votes.max() - combined_votes.min())
        
        tsObject.testData["Score"] = combined_votes
        end_time = time.time()
        return tsObject, max_time + (end_time - start_time)

    def toString(self):
        def get_abbreviation(name):
            return ''.join(char for char in name if char.isupper()).lower()

        base_learner_abbrs = sorted(set(get_abbreviation(type(learner).__name__) for learner in self.base_learners))
        
        if len(base_learner_abbrs) == 1:
            ensemble_type = f"ensemble_hom_{base_learner_abbrs[0]}"
        else:
            ensemble_type = f"ensemble_het_{'_'.join(base_learner_abbrs)}"
        
        method_name = self.method
        if self.method_params:
            param_str = '_'.join(f"{k}_{v}" for k, v in sorted(self.method_params.items()) if v is not None)
            method_name += f"_{param_str}"
        
        return f"{ensemble_type}_{method_name}"