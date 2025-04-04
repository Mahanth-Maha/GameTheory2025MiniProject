import pandas as pd
import numpy as np
import argparse
import itertools 
import time

from utils import *
from metrics import gini_coefficient, vcg_individual_rationality, vcg_budget_balance
from tqdm import tqdm


PARAMS_CSV_FILE = './data/synthetic/parameters_check.csv'

def run_single_analysis(data, params):
    results = params.copy()
    farmer_ids = data['Farmer_ID'].tolist()
    n = len(farmer_ids)
    start_time = time.time()
    gt_params = {
        'alpha': params['alpha'],
        'beta': params['beta']
    }
    
    use_exact_shapley = n <= params.get('exact_shapley_threshold', 10)
    if use_exact_shapley:
        shapley_values = shapley_value_exact(farmer_ids, data, characteristic_function_v, gt_params)
    else:
        shapley_values = shapley_value_monte_carlo(
            farmer_ids, data, characteristic_function_v, gt_params,
            n_samples=params.get('shapley_samples', 100 * n)
        )
    results['shapley_calculation_method'] = 'exact' if use_exact_shapley else 'monte_carlo'
    results['shapley_values_gini'] = gini_coefficient(list(shapley_values.values()))

    core_stable = None
    blocking_coalitions_count = None
    if n <= params.get('core_check_threshold', 15):
        try:
             char_values = get_characteristic_function_values(farmer_ids, data, characteristic_function_v, gt_params)
             core_stable, blocking_coalitions = is_in_core(shapley_values, farmer_ids, char_values)
             blocking_coalitions_count = len(blocking_coalitions) if blocking_coalitions is not None else 0
             results['core_check_performed'] = True
             results['is_shapley_in_core'] = core_stable
             results['blocking_coalitions_count'] = blocking_coalitions_count
        except Exception as e:
             print(f"[-] Core calculation failed for N={n}: {e}")
             results['core_check_performed'] = False
             results['is_shapley_in_core'] = None
             results['blocking_coalitions_count'] = None
    else:
        results['core_check_performed'] = False
        results['is_shapley_in_core'] = None
        results['blocking_coalitions_count'] = None

    vcg_winners, vcg_payments, vcg_surplus, vcg_total_payments = run_vcg_auction(data, params['vcg_price_per_credit'])
    results['vcg_num_winners'] = len(vcg_winners)
    results['vcg_total_surplus'] = vcg_surplus
    results['vcg_total_payments'] = vcg_total_payments
    results['vcg_payments_gini'] = gini_coefficient(list(vcg_payments.values()))

    results['vcg_individual_rationality_met'] = vcg_individual_rationality(vcg_payments, data)
    total_credits_allocated = data[data['Farmer_ID'].isin(vcg_winners)]['Potential_Carbon_Credits_tCO2e'].sum()
    total_buyer_value = params['vcg_price_per_credit'] * total_credits_allocated
    results['vcg_budget_balance'] = vcg_budget_balance(vcg_total_payments, total_buyer_value)
    results['vcg_efficiency_metric'] = vcg_surplus


    try:
        util_func = UTILITY_FUNCTIONS[params['utility_func_name']]
        util_params = params.get('utility_params', {})
        avg_shapley_utility = np.mean([util_func(p, util_params) for p in shapley_values.values() if p > 0])
        results['avg_shapley_utility'] = avg_shapley_utility
    except Exception as e:
        print(f"[-] Utility calculation failed: {e}")
        results['avg_shapley_utility'] = np.nan

    results['computation_time_sec'] = time.time() - start_time
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parameter search for VCM Analysis")
    parser.add_argument('-f', "--file", type=str, default='./data/synthetic/indian_farmers_carbon_market_250.csv', help="Path to the dataset file")
    parser.add_argument('-n', "--n_farmers", type=int, default=12, help="Number of farmers to sample from the dataset")
    parser.add_argument('--seed', type=int, default=24004, help="Random seed for farmer sampling")
    parser.add_argument('--params_csv_file', type=str, default=PARAMS_CSV_FILE, help="CSV file to log search results")
    args = parser.parse_args()
    
    START_SCRIPT = time.time()

    param_grid = {
        'alpha': np.linspace(0.5, 1.5, 6).tolist(),
        'beta': np.linspace(0.2, 1.0, 5).tolist(),
        'utility_func_name': ['linear', 'logarithmic'],
        'vcg_price_per_credit':  np.linspace(1000, 2500, 6).tolist(),
        # 'alpha': [1.1], # [0.8 , 0.9, 1.0, 1.1, 1.2],
        # 'beta':   [0.8], # [0.0 , 0.25, 0.5, 0.75, 1.0],
        # 'utility_func_name': ['linear'], # ['linear', 'logarithmic'],
        # 'vcg_price_per_credit': [2500], # [1500, 2000, 2500], 
        
        'shapley_samples': [1000], 
        'exact_shapley_threshold': [10], 
        'core_check_threshold': [15] 
    }
    
    print(f"[+] Parameter grid:")
    for key, values in param_grid.items():
        print(f"  [>] {key}: {values}")
    print(f"[+] Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
    
    
    
    full_data = pd.read_csv(args.file)
    if args.n_farmers < len(full_data):
            data = full_data.sample(n=args.n_farmers, random_state=args.seed)
            print(f"[>] Sampled {args.n_farmers} farmers from {args.file}")
    else:
            data = full_data
            print(f"[>] Using all {len(data)} farmers from {args.file}")
    data = data.reset_index(drop=True)
    
    keys, values = zip(*param_grid.items())
    parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []

    print(f"[+] Starting parameter search with {len(parameter_combinations)} combinations...")

    # for i, params in enumerate(tqdm(parameter_combinations, desc="Parameter Combinations")):
    for i, params in enumerate(parameter_combinations):
        print(f"[>] Running combination {i+1}/{len(parameter_combinations)}: {params}")
        try:
            params['n_farmers_analyzed'] = args.n_farmers
            params['random_seed'] = args.seed
            single_run_results = run_single_analysis(data.copy(), params)
            all_results.append(single_run_results)
        except Exception as e:
            print(f"[-] Failed combination {params}: {e}", exc_info=True)
            failed_result = params.copy()
            failed_result['status'] = 'failed'
            failed_result['error_message'] = str(e)
            # all_results.append(failed_result)


    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.params_csv_file, index=False)
    print(f"[>] Parameter search results saved to {args.params_csv_file}")
        
    results_df = pd.read_csv(args.params_csv_file)
    print(f"[>] Results DataFrame shape: {results_df.shape}")

    if not results_df.empty:
         core_checked_df = results_df[results_df['core_check_performed'] == True]
         non_core_checked_df = results_df[results_df['core_check_performed'] == False]

         best_overall = None

         if not core_checked_df.empty:
             core_stable_df = core_checked_df[core_checked_df['is_shapley_in_core'] == True]
             if not core_stable_df.empty:
                 best_core_stable = core_stable_df.loc[core_stable_df['vcg_total_surplus'].idxmax()]
                 print(f"\n--- Best Result (Core Stable & Max VCG Surplus) ---\n")
                 print(f"[>] Parameters:\n{best_core_stable[list(param_grid.keys())].to_string()}")
                 print(f"[>] Metrics:\n{best_core_stable[['vcg_total_surplus', 'shapley_values_gini', 'is_shapley_in_core']].to_string()}")
                 best_overall = best_core_stable
             else:
                  print(f"[!] No combinations resulted in Shapley allocation being in the Core.")
                  best_core_checked = core_checked_df.loc[core_checked_df['vcg_total_surplus'].idxmax()]
                  print(f"\n--- Best Result (Core Checked - Max VCG Surplus, Core NOT Stable) ---\n")
                  print(f"[>] Parameters:\n{best_core_checked[list(param_grid.keys())].to_string()}")
                  print(f"[>] Metrics:\n{best_core_checked[['vcg_total_surplus', 'shapley_values_gini', 'is_shapley_in_core']].to_string()}")
                  best_overall = best_core_checked

         if best_overall is None and not non_core_checked_df.empty :
             best_no_core_check = non_core_checked_df.loc[non_core_checked_df['vcg_total_surplus'].idxmax()]
             print(f"\n--- Best Result (Core Not Checked - Max VCG Surplus) ---\n")
             print(f"[>] Parameters:\n{best_no_core_check[list(param_grid.keys())].to_string()}")
             print(f"[>] Metrics:\n{best_no_core_check[['vcg_total_surplus', 'shapley_values_gini']].to_string()}")

         elif best_overall is None:
              print(f"[!] No successful runs found to determine best parameters")

    else:
         print(f"[!] No successful runs found in the parameter search")

    print(f"[+] Parameter search finished")

    
    END_SCRIPT = time.time()
    elapsed_time = END_SCRIPT - START_SCRIPT
    
    hours_took = elapsed_time // 3600
    minutes_took = (elapsed_time % 3600) // 60
    seconds_took = elapsed_time % 60
    print(f"[+] Total time: {int(hours_took):2d} Hrs {int(minutes_took):2d} Min {int(seconds_took):2d} Sec)")



# nohup python params.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 12 > logs/params.log 2>&1 & disown
# nohup python params.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 12 --params_csv_file ./logs/params/parameters_search.csv > logs/params/params.log 2>&1 &
# nohup python params.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 7 --params_csv_file ./logs/params/parameters_search2.csv > logs/params/params2.log 2>&1 &