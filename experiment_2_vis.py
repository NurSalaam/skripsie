import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams
from cycler import cycler

# Set up global styling
plt.style.use('default')  # Reset to default style
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['figure.titlesize'] = 16

# Monochrome color cycle
monochrome = (cycler('color', ['k']) * cycler('linestyle', ['-', '--', ':', '-.']))
rcParams['axes.prop_cycle'] = monochrome

# Function to format titles (only first letter capitalized)
def format_title(title):
    return title.capitalize()

# Function to add subfigure labels
def add_subfigure_label(ax, label):
    ax.text(-0.1, 1.1, f'({label})', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Helper function to create unique method names
def get_unique_method_name(method, params):
    if method == 'wv_ols_r2':
        return f"wv_ols_r2_{str(params.get('emphasize_diversity', False))[0]}"
    elif method == 'hard_voting':
        return f"hard_voting_{str(params.get('gaussian', False))[0]}"
    elif params:
        param_str = '_'.join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{method}_{param_str}"
    return method

# Load the data
e1_results = pd.read_csv('final_experiments/experiment_e1_results.csv')
e2_results = pd.read_csv('final_experiments/experiment_e2_results.csv')
e2_summary = pd.read_csv('final_experiments/experiment_e2_summary.csv')

# Load configurations
with open('final_experiments/experiment_e2_configurations.json', 'r') as f:
    e2_configs = json.load(f)

# Create unique method names
e2_results['unique_method'] = e2_results.apply(lambda row: get_unique_method_name(row['ensemble_method'], json.loads(row['method_params'])), axis=1)
e2_summary['unique_method'] = e2_summary.apply(lambda row: get_unique_method_name(row['ensemble_method'], json.loads(row['method_params'])), axis=1)

# Calculate relative performance for each ensemble
relative_performances = []
for config in e2_configs:
    bl_type = config['base_learner_type']
    method = config['ensemble_method']
    rep = config['repetition']
    constituent_scores = [e1_results[(e1_results['base_learner'] == bl_type) & 
                                     (e1_results['params'] == str(params))]['ucr_score'].values[0]
                          for params in config['base_learner_params']]
    avg_constituent_score = np.mean(constituent_scores)
    unique_method = get_unique_method_name(method, config['method_params'])
    ensemble_data = e2_results[(e2_results['base_learner_type'] == bl_type) & 
                               (e2_results['unique_method'] == unique_method) & 
                               (e2_results['repetition'] == rep)]
    ensemble_score = ensemble_data['ucr_score'].values[0]
    computational_time = ensemble_data['computational_time'].values[0]
    relative_score = ensemble_score / avg_constituent_score
    relative_performances.append({
        'base_learner_type': bl_type,
        'unique_method': unique_method,
        'repetition': rep,
        'relative_score': relative_score,
        'ensemble_score': ensemble_score,
        'avg_constituent_score': avg_constituent_score,
        'computational_time': computational_time
    })

relative_df = pd.DataFrame(relative_performances)



# 1. Comparative Box Plots
fig, axs = plt.subplots(2, 2, figsize=(20, 15))
base_learners = e2_results['base_learner_type'].unique()
for i, bl in enumerate(base_learners):
    ax = axs[i // 2, i % 2]
    data = []
    labels = []
    for method in relative_df['unique_method'].unique():
        data.append(relative_df[(relative_df['base_learner_type'] == bl) & 
                                (relative_df['unique_method'] == method)]['relative_score'])
        labels.append(method)
    ax.boxplot(data, labels=labels)
    ax.axhline(y=1, color='k', linestyle='--')
    ax.set_title(format_title(f'{bl} - ensemble methods'))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Relative score')
    add_subfigure_label(ax, chr(97 + i))  # 'a', 'b', 'c', 'd'

fig.suptitle(format_title('Comparative performance of ensemble methods'), fontsize=16)
plt.tight_layout()
plt.savefig('final_visualisations/experiment_2/comparative_box_plots.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. Performance Improvement Heatmap
improvement_data = relative_df.groupby(['base_learner_type', 'unique_method'])['relative_score'].mean().reset_index()
improvement_pivot = improvement_data.pivot(index='base_learner_type', columns='unique_method', values='relative_score')
improvement_pivot = (improvement_pivot - 1) * 100  # Convert to percentage improvement

fig, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(improvement_pivot, annot=True, fmt='.2f', cmap='Greys', ax=ax)

ax.set_title(format_title('Average performance improvement of ensemble methods (%)'))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
add_subfigure_label(ax, 'a')

plt.tight_layout()
plt.savefig('final_visualisations/experiment_2/performance_improvement_heatmap.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.close()

# # 3. Reliability Improvement Bar Chart
# plt.figure(figsize=(20, 12))
# base_learners = e1_results['base_learner'].unique()
# ensemble_methods = relative_df['unique_method'].unique()

# x = np.arange(len(base_learners))
# width = 0.8 / (len(ensemble_methods) + 1)  # +1 for the individual learner bar

# e1_cv = e1_results.groupby('base_learner')['ucr_score'].apply(lambda x: x.std() / x.mean())
# plt.bar(x, e1_cv, width, label='Individual')

# for i, method in enumerate(ensemble_methods, 1):
#     cv_scores = []
#     for bl in base_learners:
#         cv = relative_df[(relative_df['base_learner_type'] == bl) & 
#                          (relative_df['unique_method'] == method)]['relative_score'].std() / \
#              relative_df[(relative_df['base_learner_type'] == bl) & 
#                          (relative_df['unique_method'] == method)]['relative_score'].mean()
#         cv_scores.append(cv)
#     plt.bar(x + width * i, cv_scores, width, label=method)

# plt.xlabel('Base Learner Type')
# plt.ylabel('Coefficient of Variation')
# plt.title('Reliability Comparison: Individual vs Ensemble Methods')
# plt.xticks(x + width * (len(ensemble_methods) / 2), base_learners)
# plt.legend(loc='upper left', bbox_to_anchor=(1,1))
# plt.tight_layout()
# plt.savefig('final_visualisations/experiment_2/reliability_improvement_bar_chart.png')
# plt.close()

# 4. Computational Time vs. Performance Scatter Plot
# plt.figure(figsize=(15, 10))
# for bl in base_learners:
#     e1_data = e1_results[e1_results['base_learner'] == bl]
#     plt.scatter(e1_data['computational_time'], e1_data['ucr_score'], 
#                 label=f'{bl} (Individual)', alpha=0.7, marker='o')
    
#     e2_data = relative_df[relative_df['base_learner_type'] == bl]
#     plt.scatter(e2_data['computational_time'], e2_data['relative_score'], 
#                 label=f'{bl} (Ensemble)', alpha=0.7, marker='^')

# plt.xlabel('Computational Time')
# plt.ylabel('UCR Score / Relative Score')
# plt.title('Performance vs Computational Time: Individual vs Ensemble Methods')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.savefig('final_visualisations/experiment_2/performance_vs_time_scatter.png')
# plt.close()

# # 5. Performance Relative to Constituent Learners (Violin Plot)
# plt.figure(figsize=(20, 12))
# sns.violinplot(x='unique_method', y='relative_score', hue='base_learner_type', 
#                data=relative_df, split=True)
# plt.axhline(y=1, color='r', linestyle='--')
# plt.title('Ensemble Performance Relative to Constituent Learners')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('final_visualisations/experiment_2/relative_performance_violin.png')
# plt.close()

print("All visualisations have been created and saved in the 'final_visualisations/experiment_2' directory.")