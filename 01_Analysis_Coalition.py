import pandas as pd
import numpy as np
import argparse
import itertools
import os
import matplotlib.pyplot as plt
import seaborn as sns

import best_config
from utils import *
from metrics import gini_coefficient

def compare_individual_vs_coalition(data, params, sample_n, seed):
    print(f"\n--- Running Individual vs. Grand Coalition (N={sample_n}) ---\n")
    if sample_n > len(data):
        sample_n = len(data)
    sample_data = data.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    farmer_ids = sample_data['Farmer_ID'].tolist()

    gt_params = {'alpha': params['alpha'], 'beta': params['beta'], 'gamma': params.get('gamma', 0.0)}

    use_exact = sample_n <= params.get('exact_shapley_threshold', 10)
    if use_exact:
        shapley_values = shapley_value_exact(farmer_ids, sample_data, characteristic_function_v, gt_params)
    else:
        shapley_values = shapley_value_monte_carlo(
            farmer_ids, sample_data, characteristic_function_v, gt_params,
            n_samples=params.get('shapley_samples', 100 * sample_n)
        )

    results = []
    for farmer_id in farmer_ids:
        standalone = sample_data.loc[sample_data['Farmer_ID'] == farmer_id, 'Standalone_Payoff_INR'].iloc[0]
        shapley = shapley_values.get(farmer_id, 0)
        gain_shapley = shapley - standalone
        results.append({
            'Farmer_ID': farmer_id,
            'Standalone_Payoff': standalone,
            'Shapley_Value': shapley,
            'Gain_vs_Standalone': gain_shapley
        })

    results_df = pd.DataFrame(results)
    avg_gain = results_df['Gain_vs_Standalone'].mean()
    pct_better_off = (results_df['Gain_vs_Standalone'] > 0).mean() * 100

    print(f"[>] Average Gain (Shapley vs Standalone): {avg_gain:.2f} INR")
    print(f"[>] Percentage of Farmers Better Off: {pct_better_off:.1f}%")
    print(results_df.round(2))
    return results_df

def analyze_coalition_size(data, params, sizes_to_test, num_samples_per_size, seed):
    print(f"\n--- Running Coalition Size Analysis (Sizes: {sizes_to_test}) ---\n")
    all_size_results = []
    base_seed = seed

    for size in sizes_to_test:
        if size > len(data):
             print(f"[!] Skipping size {size}, larger than dataset {len(data)}")
             continue
        print(f"[>] Analyzing size {size}...")
        size_results_v_per_farmer = []
        size_results_avg_shapley = []

        for i in range(num_samples_per_size):
            current_seed = base_seed + i * len(sizes_to_test) + sizes_to_test.index(size)
            sample_data = data.sample(n=size, random_state=current_seed).reset_index(drop=True)
            farmer_ids = sample_data['Farmer_ID'].tolist()

            gt_params = {'alpha': params['alpha'], 'beta': params['beta'], 'gamma': params.get('gamma', 0.0)}

            v_s = characteristic_function_v(farmer_ids, sample_data, **gt_params)
            size_results_v_per_farmer.append(v_s / size if size > 0 else 0)

            use_exact = size <= params.get('exact_shapley_threshold', 10)
            if use_exact:
                shapley_values = shapley_value_exact(farmer_ids, sample_data, characteristic_function_v, gt_params)
            else:
                shapley_values = shapley_value_monte_carlo(
                    farmer_ids, sample_data, characteristic_function_v, gt_params,
                    n_samples=params.get('shapley_samples', 100 * size)
                )
            avg_shapley = np.mean(list(shapley_values.values())) if shapley_values else 0
            size_results_avg_shapley.append(avg_shapley)

        all_size_results.append({
            'Size': size,
            'Avg_Value_per_Farmer': np.mean(size_results_v_per_farmer),
            'StdDev_Value_per_Farmer': np.std(size_results_v_per_farmer),
            'Avg_Shapley_Value': np.mean(size_results_avg_shapley),
            'StdDev_Shapley_Value': np.std(size_results_avg_shapley)
        })

    results_df = pd.DataFrame(all_size_results)
    print(f"[>] Coalition Size Analysis Results:")
    print(results_df.round(2))
    return results_df

def analyze_distribution_stability(data, params, sample_n, seed):
    print(f"\n--- Running Distribution Stability Analysis (N={sample_n}) ---\n")
    if sample_n > len(data):
        sample_n = len(data)
    if sample_n > params.get('core_check_threshold', 15):
        print(f"[!] N={sample_n} exceeds Core check threshold ({params.get('core_check_threshold', 15)}). Skipping stability check.")
        return None

    sample_data = data.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    farmer_ids = sample_data['Farmer_ID'].tolist()
    n = len(farmer_ids)

    gt_params = {'alpha': params['alpha'], 'beta': params['beta'], 'gamma': params.get('gamma', 0.0)}

    try:
         char_values = get_characteristic_function_values(farmer_ids, sample_data, characteristic_function_v, gt_params)
         grand_coalition_key = tuple(sorted(farmer_ids))
         v_n = char_values.get(grand_coalition_key, 0)
         if v_n == 0 and n > 0: print(f"[!] Grand coalition value is 0.")
    except Exception as e:
         print(f"[-] Failed to calculate characteristic values: {e}")
         return None

    stability_results = []

    # 1. Shapley Value Allocation
    use_exact = n <= params.get('exact_shapley_threshold', 10)
    if use_exact:
        shapley_values = shapley_value_exact(farmer_ids, sample_data, characteristic_function_v, gt_params)
    else:
        shapley_values = shapley_value_monte_carlo(
            farmer_ids, sample_data, characteristic_function_v, gt_params,
            n_samples=params.get('shapley_samples', 100 * n)
        )
    is_stable_shapley, _ = is_in_core(shapley_values, farmer_ids, char_values)
    gini_shapley = gini_coefficient(list(shapley_values.values()))
    stability_results.append({'Rule': 'Shapley', 'Is_in_Core': is_stable_shapley, 'Gini': gini_shapley})

    # 2. Equal Split Allocation
    equal_split_val = v_n / n if n > 0 else 0
    equal_split_alloc = {f_id: equal_split_val for f_id in farmer_ids}
    is_stable_equal, _ = is_in_core(equal_split_alloc, farmer_ids, char_values)
    gini_equal = gini_coefficient(list(equal_split_alloc.values())) # Should be 0
    stability_results.append({'Rule': 'Equal Split', 'Is_in_Core': is_stable_equal, 'Gini': gini_equal})

    # 3. Proportional Split Allocation (based on Standalone Payoff)
    total_standalone = sample_data['Standalone_Payoff_INR'].sum()
    prop_alloc = {}
    if total_standalone > 0:
        for idx, row in sample_data.iterrows():
            prop_alloc[row['Farmer_ID']] = (row['Standalone_Payoff_INR'] / total_standalone) * v_n
    else: 
        prop_alloc = {f_id: equal_split_val for f_id in farmer_ids}

    is_stable_prop, _ = is_in_core(prop_alloc, farmer_ids, char_values)
    gini_prop = gini_coefficient(list(prop_alloc.values()))
    stability_results.append({'Rule': 'Proportional (Standalone)', 'Is_in_Core': is_stable_prop, 'Gini': gini_prop})

    results_df = pd.DataFrame(stability_results)
    print(f"[>] Distribution Stability Analysis Results:")
    print(results_df.round(4))
    return results_df

def plot_coalition_size_results(df, output_dir):
    if df is None or df.empty: return
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.errorbar(df['Size'], df['Avg_Value_per_Farmer'], yerr=df['StdDev_Value_per_Farmer'], fmt='-o', capsize=5)
    plt.xlabel("Coalition Size (S)")
    plt.ylabel("Average Value per Farmer (v(S)/S)")
    plt.title("Coalition Value per Farmer vs. Size")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.errorbar(df['Size'], df['Avg_Shapley_Value'], yerr=df['StdDev_Shapley_Value'], fmt='-o', capsize=5, color='orange')
    plt.xlabel("Coalition Size (S)")
    plt.ylabel("Average Shapley Value (within coalition)")
    plt.title("Average Shapley Value vs. Size")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'coalition_size_analysis.png')
    plt.savefig(filename)
    print(f"[>] Saved coalition size plot: {filename}")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Coalition Formation Analysis Experiments")
    parser.add_argument('-f', "--file", type=str, default=best_config.DEFAULT_DATA_FILE, help="Path to the dataset file.")
    parser.add_argument('-n', "--n_farmers", type=int, default=best_config.N_COALITION_ANALYSIS, help="Number of farmers to sample.")
    parser.add_argument('--seed', type=int, default=best_config.RANDOM_SEED, help="Random seed for sampling.")
    parser.add_argument('--params_alpha', type=float, default=best_config.BASELINE_PARAMS['alpha'], help="Alpha parameter.")
    parser.add_argument('--params_beta', type=float, default=best_config.BASELINE_PARAMS['beta'], help="Beta parameter.")
    parser.add_argument('--params_gamma', type=float, default=best_config.BASELINE_PARAMS['gamma'], help="Gamma parameter (cost sharing).")
    parser.add_argument('--plot_dir', type=str, default=os.path.join(best_config.PLOT_OUTPUT_DIR, 'coalition/'), help="Directory for saving plots.")
    args = parser.parse_args()

    args.plot_dir = os.path.join(args.plot_dir, f"n_farmers_{args.n_farmers}/")

    os.makedirs(args.plot_dir, exist_ok=True)

    try:
        data = pd.read_csv(args.file)
        print(f"[>] Loaded data from {args.file}")
    except FileNotFoundError:
        print(f"[-] Data file not found: {args.file}")
        exit()

    current_params = best_config.BASELINE_PARAMS.copy()
    current_params['alpha'] = args.params_alpha
    current_params['beta'] = args.params_beta
    current_params['gamma'] = args.params_gamma

    individual_vs_coalition_results = compare_individual_vs_coalition(data, current_params, args.n_farmers, args.seed)

    coalition_sizes_to_test = sorted(list(set([3, 5, 8, args.n_farmers])))
    size_analysis_results = analyze_coalition_size(data, current_params, coalition_sizes_to_test, num_samples_per_size=20, seed=args.seed)
    plot_coalition_size_results(size_analysis_results, args.plot_dir)

    distribution_stability_results = analyze_distribution_stability(data, current_params, args.n_farmers, args.seed)
 
# python 01_Analysis_Coalition.py --n_farmers 12 --params_alpha 1.25 --params_beta 0.0 --plot_dir ./logs/plots/ > ./logs/coalition/CA_n_12.log
# nohup python 01_Analysis_Coalition.py --n_farmers 12 --params_alpha 1.25 --params_beta 0.0 --plot_dir ./logs/plots/ > ./logs/coalition/CA_n_12.log 2&>1 & disown