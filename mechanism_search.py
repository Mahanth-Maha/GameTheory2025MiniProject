import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
import argparse

from generate_data import get_data 
from mechanism import run_vcg_auction_mechanism, run_uniform_price_auction_mechanism, run_shapley_allocation_mechanism
from metrics import gini_coefficient, check_individual_rationality
from plot_mechanism import plot_mechanism_comparison
from utils import characteristic_function_v 

DEFAULT_N_FARMERS = 12
DEFAULT_SEED = 24004
DEFAULT_DATA_FILE = './data/synthetic/indian_farmers_carbon_market_250.csv'
DEFAULT_OUTPUT_CSV = './data/synthetic/mechanism_comparison_n_12.csv'
DEFAULT_PLOT_DIR = './logs/plots/mechanism_search'

DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.0

DEFAULT_VCG_PRICE = 2500
DEFAULT_BUYER_DEMAND_CREDITS = 15

K = 9

VARYING_PARAM_NAME = 'market_price_per_credit'
PARAM_VALUES = np.linspace(1000, 3100, K)

MECHANISMS_TO_RUN = [
    'VCG Auction',
    'Uniform Price Auction',
    'Shapley Allocation'
]
SHAPLEY_SAMPLES = 1000
CORE_CHECK_THRESHOLD = 15

def calculate_ir_percentage(payoffs, data):
    n_farmers = len(data)
    if n_farmers == 0:
        return 100.0

    standalone_payoffs = data.set_index('Farmer_ID')['Standalone_Payoff_INR'].to_dict()
    met_ir_count = 0
    tolerance = 1e-6

    for farmer_id, payoff in payoffs.items():
         if farmer_id in standalone_payoffs:
              if payoff >= standalone_payoffs[farmer_id] - tolerance:
                   met_ir_count += 1

    return (met_ir_count / n_farmers) * 100 if n_farmers > 0 else 100.0

def run_single_iteration(data, params):
    iteration_results = []
    n_farmers = len(data)
    all_farmer_ids = data['Farmer_ID'].tolist()

    if 'Standalone_Payoff_INR' not in data.columns:
        raise ValueError("Dataframe missing 'Standalone_Payoff_INR' for IR checks.")

    current_market_price = params[VARYING_PARAM_NAME]
    print(f"[>] Running Mechanisms with {VARYING_PARAM_NAME}={current_market_price:.2f} and params: {params}")


    if 'VCG Auction' in MECHANISMS_TO_RUN:
        try:
            # print(f"[+] Running VCG...")
            vcg_result = run_vcg_auction_mechanism(data, price_per_credit=current_market_price)
            # print(f"[*] VCG Result: {vcg_result}")
            ir_met, _ = check_individual_rationality(vcg_result['payoffs'], data)
            ir_perc = calculate_ir_percentage(vcg_result['payoffs'], data)
            avg_profit = np.mean(list(vcg_result['payoffs'].values())) if vcg_result['payoffs'] else 0.0
            gini = gini_coefficient(list(vcg_result['payoffs'].values()))

            iteration_results.append({
                **params, 
                'mechanism_name': vcg_result['mechanism_name'], 
                'avg_farmer_profit': avg_profit,
                'gini_coefficient': gini,
                'ir_met_overall': ir_met,
                'ir_met_percentage': ir_perc,           
                'num_winners': len(vcg_result['winners']),
                'buyer_cost': vcg_result['buyer_cost'],
                'total_surplus': vcg_result.get('total_surplus', np.nan),
                'is_in_core': None,
                'core_check_performed': False,
            })
            # print("[+] VCG Appended.")
        except Exception as e:
            print(f"\n[ERROR] VCG Auction failed for params {params}: {e}")
            iteration_results.append({**params, 'mechanism_name': 'VCG Auction', 'status': 'failed', 'error': str(e)})


    if 'Uniform Price Auction' in MECHANISMS_TO_RUN:
         try:
            # print(f"[+] Running Uniform Price...")
            demand = params.get('buyer_demand', DEFAULT_BUYER_DEMAND_CREDITS)
            up_result = run_uniform_price_auction_mechanism(data, buyer_total_credits_demand=demand)
            # print(f"[*] UP Result: {up_result}")
            ir_met, _ = check_individual_rationality(up_result['payoffs'], data)
            ir_perc = calculate_ir_percentage(up_result['payoffs'], data)
            avg_profit = np.mean(list(up_result['payoffs'].values())) if up_result['payoffs'] else 0.0
            gini = gini_coefficient(list(up_result['payoffs'].values()))

            iteration_results.append({
                **params,
                'mechanism_name': up_result['mechanism_name'],
                'avg_farmer_profit': avg_profit,
                'gini_coefficient': gini,
                'ir_met_overall': ir_met,
                'ir_met_percentage': ir_perc,         
                'num_winners': len(up_result['winners']),
                'buyer_cost': up_result['buyer_cost'],
                'total_surplus': np.nan,
                'is_in_core': None,
                'core_check_performed': False,
            })
            # print("[+] Uniform Price Appended.")
         except Exception as e:
            print(f"\n[ERROR] Uniform Price Auction failed for params {params}: {e}")
            iteration_results.append({**params, 'mechanism_name': 'Uniform Price Auction', 'status': 'failed', 'error': str(e)})


    if 'Shapley Allocation' in MECHANISMS_TO_RUN:
        try:
            # print(f"[+] Running Shapley...")
            gt_params_shapley = {'alpha': params['alpha'], 'beta': params['beta']}
            v_N = characteristic_function_v(all_farmer_ids, data, **gt_params_shapley) if all_farmer_ids else 0.0

            shapley_result = run_shapley_allocation_mechanism(
                data,
                market_price_per_credit=current_market_price,
                alpha=params['alpha'],
                beta=params['beta'],
                use_exact_shapley=(n_farmers <= 10),
                shapley_samples=SHAPLEY_SAMPLES,
                core_check_threshold=CORE_CHECK_THRESHOLD
            )
            # print(f"[*] Shapley Result: {shapley_result}")
            ir_met, _ = check_individual_rationality(shapley_result['payoffs'], data)
            ir_perc = calculate_ir_percentage(shapley_result['payoffs'], data)
            avg_profit = np.mean(list(shapley_result['payoffs'].values())) if shapley_result['payoffs'] else 0.0
            gini = gini_coefficient(list(shapley_result['payoffs'].values()))

            iteration_results.append({
                **params,
                'mechanism_name': shapley_result['mechanism_name'],
                'avg_farmer_profit': avg_profit,
                'gini_coefficient': gini,
                'ir_met_overall': ir_met,
                'ir_met_percentage': ir_perc,
                'num_winners': n_farmers,
                'buyer_cost': np.nan,
                'total_surplus': v_N,
                'is_in_core': shapley_result['is_in_core'],
                'core_check_performed': shapley_result['core_check_performed'],
            })
            # print("[+] Shapley Appended.")
        except Exception as e:
            print(f"\n[ERROR] Shapley Allocation failed for params {params}: {e}")
            iteration_results.append({**params, 'mechanism_name': 'Shapley Allocation', 'status': 'failed', 'error': str(e)})


    return iteration_results

def find_best_mechanism(results_df):
    if results_df.empty:
        print("[!] Results DataFrame is empty. Cannot find best mechanism.")
        return None, None

    if results_df.empty:
        print("[!] No successful runs found after filtering. Cannot find best mechanism.")
        return None, None

    ir_summary = results_df.groupby('mechanism_name')['ir_met_percentage'].mean()

    reliable_threshold = 80.0
    reliable_mechanisms = ir_summary[ir_summary >= reliable_threshold].index.tolist()

    print(f"\n--- Best Mechanism Analysis ---")
    print(f"Criteria: Reliability (Avg IR % >= {reliable_threshold}%), Max Avg Profit, Min Gini")
    print(f"Avg IR % per mechanism (based on successful runs):\n{ir_summary.round(2)}")

    if not reliable_mechanisms:
        print(f"\n[!] No mechanism consistently met the IR threshold ({reliable_threshold}%).")
        if 'avg_farmer_profit' in results_df.columns:
             best_avg_profit_row = results_df.loc[results_df['avg_farmer_profit'].idxmax()]
             best_mech_name = best_avg_profit_row['mechanism_name']
             print(f"Falling back to mechanism with highest single avg profit: {best_mech_name}")
             best_mech_row = best_avg_profit_row

             print(f"\nBest Overall Mechanism (Fallback - Max Avg Profit): {best_mech_name}")
             print(f"Performance (at specific parameters where max profit occurred):")
             print(best_mech_row[[VARYING_PARAM_NAME,'avg_farmer_profit', 'gini_coefficient', 'ir_met_percentage', 'buyer_cost', 'is_in_core']])
             return best_mech_name, best_mech_row
        else:
             print("[-] Cannot determine fallback best mechanism without 'avg_farmer_profit'.")
             return None, None


    print(f"Reliable Mechanisms meeting IR threshold: {reliable_mechanisms}")
    filtered_results = results_df[results_df['mechanism_name'].isin(reliable_mechanisms)]

    avg_profit_summary = filtered_results.groupby('mechanism_name')['avg_farmer_profit'].mean()

    if avg_profit_summary.empty:
         print("[-] Could not calculate average profit for reliable mechanisms.")
         return None, None

    best_profit_mech = avg_profit_summary.idxmax()
    best_profit_value = avg_profit_summary.max()
    print(f"Avg Profit for reliable mechanisms:\n{avg_profit_summary.round(2)}")


    candidates = avg_profit_summary[avg_profit_summary >= 0.95 * best_profit_value].index.tolist()

    if len(candidates) == 1:
        best_mech_name = candidates[0]
        print(f"Mechanism '{best_mech_name}' has highest average profit among reliable options.")
    else:
        print(f"Multiple candidates with similar high profit: {candidates}. Breaking tie with Gini...")
        gini_summary = filtered_results[filtered_results['mechanism_name'].isin(candidates)].groupby('mechanism_name')['gini_coefficient'].mean()
        if gini_summary.empty:
             print("[!] Could not calculate Gini for tie-breaking. Selecting first candidate.")
             best_mech_name = candidates[0]
        else:
            best_mech_name = gini_summary.idxmin()
            print(f"Avg Gini for candidates:\n{gini_summary.round(4)}")
            print(f"Tie broken: '{best_mech_name}' selected due to lower average Gini.")


    median_param_value = np.median(results_df[VARYING_PARAM_NAME].unique())
    best_mech_df = results_df[results_df['mechanism_name'] == best_mech_name]

    if best_mech_df.empty:
         print(f"[!] No data found for the best mechanism '{best_mech_name}' to select representative row.")
         return best_mech_name, None 
    closest_param_idx = (best_mech_df[VARYING_PARAM_NAME] - median_param_value).abs().idxmin()
    best_mech_row = best_mech_df.loc[closest_param_idx]


    print(f"\nBest Overall Reliable Mechanism: {best_mech_name}")
    print(f"Showing performance near median '{VARYING_PARAM_NAME}' ({best_mech_row[VARYING_PARAM_NAME]:.2f}):")
    display_cols = [VARYING_PARAM_NAME,'avg_farmer_profit', 'gini_coefficient', 'ir_met_percentage']
    if 'buyer_cost' in best_mech_row.index and not pd.isna(best_mech_row['buyer_cost']):
        display_cols.append('buyer_cost')
    if 'is_in_core' in best_mech_row.index and 'core_check_performed' in best_mech_row.index and best_mech_row['core_check_performed'] == True and not pd.isna(best_mech_row['is_in_core']):
         display_cols.extend(['is_in_core', 'core_check_performed'])
    print(best_mech_row[display_cols].to_string())


    return best_mech_name, best_mech_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with VCM Mechanisms")
    parser.add_argument('-f', "--file", type=str, default=DEFAULT_DATA_FILE, help="Path to farmer dataset CSV")
    parser.add_argument('-n', "--n_farmers", type=int, default=DEFAULT_N_FARMERS, help="Number of farmers to sample")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument('--output_csv', type=str, default=DEFAULT_OUTPUT_CSV, help="Output CSV file for results")
    parser.add_argument('--plot_dir', type=str, default=DEFAULT_PLOT_DIR, help="Directory to save plots")
    args = parser.parse_args()

    start_time = time.time()

    try:
        full_data = pd.read_csv(args.file)
        if args.n_farmers < len(full_data):
            data = full_data.sample(n=args.n_farmers, random_state=args.seed)
            print(f"[+] Sampled {args.n_farmers} farmers from {args.file}")
        else:
            data = full_data
            print(f"[+] Using all {len(data)} farmers from {args.file}")
        data = data.reset_index(drop=True)
        if 'Standalone_Payoff_INR' not in data.columns:
             raise ValueError("Dataset must contain 'Standalone_Payoff_INR' column for IR checks.")

    except FileNotFoundError:
        print(f"[!] Data file {args.file} not found. Generating synthetic data.")
        data = get_data(n_samples=args.n_farmers, seeding=args.seed)
        if 'Standalone_Payoff_INR' not in data.columns:
             data['Standalone_Payoff_INR'] = np.random.normal(10000, 2000, args.n_farmers).clip(lower=1000)
             print("[!] Added placeholder 'Standalone_Payoff_INR' to generated data.")
    except ValueError as ve:
         print(f"[-] Data loading failed: {ve}")
         exit()
         
    all_results = []
    print(f"[+] Starting mechanism comparison across {K} values of '{VARYING_PARAM_NAME}'...")

    base_params = {
        'alpha': DEFAULT_ALPHA,
        'beta': DEFAULT_BETA,
        'n_farmers_analyzed': args.n_farmers,
        'seed': args.seed,
        'buyer_demand': DEFAULT_BUYER_DEMAND_CREDITS
    }

    temp_results_list = []
    for param_val in tqdm(PARAM_VALUES, desc="Parameter Iterations"):
        current_params = base_params.copy()
        current_params[VARYING_PARAM_NAME] = param_val
        try:
            iteration_results = run_single_iteration(data.copy(), current_params)
            temp_results_list.extend(iteration_results)
        except Exception as e:
            print(f"\n[---*---] Iteration failed for {VARYING_PARAM_NAME}={param_val}. Params: {current_params}. Error: {e}")
            temp_results_list.append({**current_params, 'mechanism_name': 'Iteration Failed', 'status': 'iteration_failed', 'error': str(e)})


    if not temp_results_list:
         print("[-] No results were generated from any iteration.")
         results_df = pd.DataFrame()
    else:
         results_df = pd.DataFrame(temp_results_list)


    if not results_df.empty:
        if 'is_in_core' in results_df.columns and 'core_check_performed' in results_df.columns :
            results_df['is_in_core_numeric'] = results_df.apply(lambda row: 1 if row['core_check_performed'] and row['is_in_core'] is True else 0 if row['core_check_performed'] and row['is_in_core'] is False else pd.NA,axis=1).astype('Int64')
        else:
            print("[!] 'is_in_core' or 'core_check_performed' column missing, cannot create numeric core column.")

        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n[+] Experiment results saved to {args.output_csv}")

        best_mech_name, best_mech_row = find_best_mechanism(results_df)

        plot_successful_runs_df = results_df.copy()

        if not plot_successful_runs_df.empty:
             required_plot_cols = ['mechanism_name', VARYING_PARAM_NAME, 'avg_farmer_profit', 'gini_coefficient', 'ir_met_percentage']
             if all(col in plot_successful_runs_df.columns for col in required_plot_cols):
                 plot_mechanism_comparison(plot_successful_runs_df, VARYING_PARAM_NAME, output_dir=args.plot_dir)
             else:
                 print("[!] Missing one or more required columns for plotting. Skipping plots.")
                 print(f"  Required: {required_plot_cols}")
                 print(f"  Available: {plot_successful_runs_df.columns.tolist()}")
        else:
             print("[!] No successful runs to plot.")

    else:
        print("[!] No results generated to save or analyze.")


    end_time = time.time()
    print(f"[+] Experiment finished in {end_time - start_time:.2f} seconds.")
    
# nohup python mechanism_search.py -n 12 --output_csv ./data/synthetic/mechanism_comparison_n_12.csv  > logs/mechanism_search/search_n_12.log &
# nohup python mechanism_search.py -n 25 --output_csv ./data/synthetic/mechanism_comparison_n_25.csv  > logs/mechanism_search/search_n_25.log &

# nohup python mechanism_search.py -n 12 --output_csv ./data/synthetic/mechanism_comparison_n_12.csv  > logs/mechanism_search/search_n_12.log &
# nohup python mechanism_search.py -n 25 --output_csv ./data/synthetic/mechanism_comparison_n_25.csv  > logs/mechanism_search/search_n_25.log &
# nohup python mechanism_search.py -n 100 --output_csv ./data/synthetic/mechanism_comparison_n_100.csv  > logs/mechanism_search/search_n_100.log &
# nohup python mechanism_search.py -n 250 --output_csv ./data/synthetic/mechanism_comparison_n_250.csv  > logs/mechanism_search/search_n_250.log &
