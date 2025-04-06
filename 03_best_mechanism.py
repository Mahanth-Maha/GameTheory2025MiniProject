import pandas as pd
import numpy as np
import time
import argparse

from utils import *
from metrics import gini_coefficient,check_individual_rationality, calculate_ir_percentage
from generate_data import get_data

BEST_ALPHA = 1.25
BEST_BETA = 0.0
DEFAULT_DATA_FILE = './data/synthetic/indian_farmers_carbon_market_250.csv'
DEFAULT_N_FARMERS = 100
RANDOM_SEED = 24004
SHAPLEY_SAMPLES = 10000
CORE_CHECK_THRESHOLD = 15
EXACT_SHAPLEY_THRESHOLD = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Final Analysis with Best Mechanism (Shapley)")
    parser.add_argument('-f', "--file", type=str, default=DEFAULT_DATA_FILE, help="Path to farmer dataset CSV")
    parser.add_argument('-n', "--n_farmers", type=int, default=DEFAULT_N_FARMERS, help="Number of farmers to analyze")
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Random seed for sampling")
    parser.add_argument('--alpha', type=float, default=BEST_ALPHA, help="Alpha parameter override")
    parser.add_argument('--beta', type=float, default=BEST_BETA, help="Beta parameter override")
    args = parser.parse_args()

    start_time = time.time()

    try:
        full_data = pd.read_csv(args.file)
        if args.n_farmers < len(full_data):
            data = full_data.sample(n=args.n_farmers, random_state=args.seed)
            print(f"[+] Sampled {args.n_farmers} farmers from {args.file} (Seed: {args.seed})")
        else:
            args.n_farmers = len(full_data)
            data = full_data
            print(f"[+] Using all {args.n_farmers} farmers from {args.file}")
        data = data.reset_index(drop=True)
        if 'Standalone_Payoff_INR' not in data.columns:
             raise ValueError("Dataset must contain 'Standalone_Payoff_INR'.")
        if 'True_Cost_per_Credit_INR' not in data.columns:
             raise ValueError("Dataset must contain 'True_Cost_per_Credit_INR'.")
        if 'Potential_Carbon_Credits_tCO2e' not in data.columns:
             raise ValueError("Dataset must contain 'Potential_Carbon_Credits_tCO2e'.")

    except FileNotFoundError:
        print(f"[Error] Data file {args.file} not found.")
        exit()
    except ValueError as ve:
         print(f"[Error] Data loading failed: {ve}")
         exit()

    n_farmers = args.n_farmers
    farmer_ids = data['Farmer_ID'].tolist()
    gt_params = {'alpha': args.alpha, 'beta': args.beta}

    print("\n--- Running Final Analysis ---")
    print(f"Mechanism: Shapley Allocation")
    print(f"Parameters: alpha={gt_params['alpha']}, beta={gt_params['beta']}")
    print(f"Number of Farmers (N): {n_farmers}")


    use_exact = n_farmers <= EXACT_SHAPLEY_THRESHOLD
    print(f"Calculating Shapley Values ({'Exact' if use_exact else 'Monte Carlo'})...")
    if use_exact:
        farmer_payoffs = shapley_value_exact(farmer_ids, data, characteristic_function_v, gt_params)
    else:
        farmer_payoffs = shapley_value_monte_carlo(
            farmer_ids, data, characteristic_function_v, gt_params,
            n_samples=SHAPLEY_SAMPLES
        )
    print("Shapley calculation complete.")

    print("Calculating Metrics...")
    avg_profit = np.mean(list(farmer_payoffs.values())) if farmer_payoffs else 0.0
    gini = gini_coefficient(list(farmer_payoffs.values()))
    ir_met_overall, failing_ir_farmers = check_individual_rationality(farmer_payoffs, data)
    ir_perc = calculate_ir_percentage(farmer_payoffs, data)
    v_n = characteristic_function_v(farmer_ids, data, **gt_params) if farmer_ids else 0.0


    core_status = "Not Checked (N > Threshold)"
    core_check_performed = False
    if n_farmers <= CORE_CHECK_THRESHOLD:
        print(f"Checking Core Stability (N={n_farmers})...")
        try:
            char_values = get_characteristic_function_values(farmer_ids, data, characteristic_function_v, gt_params)
            is_stable, blocking_coalitions = is_in_core(farmer_payoffs, farmer_ids, char_values)
            core_status = "Stable (In Core)" if is_stable else "Unstable (Not in Core)"
            if not is_stable:
                core_status += f" - {len(blocking_coalitions)} blocking coalition(s)"
            core_check_performed = True
            print("Core check complete.")
        except Exception as e:
            core_status = f"Check Failed ({e})"
            print(f"[Error] Core check failed: {e}")
    else:
        print(f"Core check skipped (N={n_farmers} > {CORE_CHECK_THRESHOLD})")

    print("\n--- Final Run Results ---")
    print(f"Mechanism:                  Shapley Allocation")
    print(f"Parameters (alpha, beta): ({gt_params['alpha']}, {gt_params['beta']})")
    print(f"Number of Farmers (N):      {n_farmers}")
    print("-" * 30)
    print(f"Average Farmer Profit:    {avg_profit:,.2f} INR")
    print(f"Fairness (Gini Coeff.):   {gini:.4f} (0=Perfect Equality)")
    print(f"Individual Rationality:   {'Met by ALL' if ir_met_overall else f'FAILED for {len(failing_ir_farmers)} farmer(s)'}")
    print(f"IR Percentage Met:        {ir_perc:.1f}%")
    print(f"Core Stability:           {core_status}")
    print(f"Total Value Dist. (v(N)): {v_n:,.2f} INR")
    print("-" * 30)

    if farmer_payoffs:
        payoff_df = pd.DataFrame(farmer_payoffs.items(), columns=['Farmer_ID', 'Shapley_Payoff'])
        payoff_df = payoff_df.merge(data[['Farmer_ID', 'Standalone_Payoff_INR']], on='Farmer_ID')
        payoff_df['Gain'] = payoff_df['Shapley_Payoff'] - payoff_df['Standalone_Payoff_INR']
        print("\nSample Farmer Payoffs:")
        print(payoff_df.head(10).round(2).to_string(index=False))

    end_time = time.time()
    print(f"\n[+] Final run finished in {end_time - start_time:.2f} seconds.")
    
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv > logs/best_mechanism.log 2>&1 &
# tail -f logs/best_mechanism.log

# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.0 --beta 0.0 > logs/best_mechanism_a_1.0_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.0 --beta 0.1 > logs/best_mechanism_a_1.0_b_0.1.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.0 --beta 0.2 > logs/best_mechanism_a_1.0_b_0.2.log 2>&1 &

# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.5 --beta 0.0 > logs/best_mechanism_a_0.5_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.5 --beta 0.1 > logs/best_mechanism_a_0.5_b_0.1.log 2>&1 &  
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.5 --beta 0.2 > logs/best_mechanism_a_0.5_b_0.2.log 2>&1 &

# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.75 --beta 0.0 > logs/best_mechanism_a_0.75_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.75 --beta 0.1 > logs/best_mechanism_a_0.75_b_0.1.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 0.75 --beta 0.2 > logs/best_mechanism_a_0.75_b_0.2.log 2>&1 &

# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.25 --beta 0.0 > logs/best_mechanism_a_1.25_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.25 --beta 0.1 > logs/best_mechanism_a_1.25_b_0.1.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.25 --beta 0.2 > logs/best_mechanism_a_1.25_b_0.2.log 2>&1 &


# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.05 --beta 0.0 > logs/best_mechanism_a_1.05_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.10 --beta 0.0 > logs/best_mechanism_a_1.10_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.15 --beta 0.0 > logs/best_mechanism_a_1.15_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.20 --beta 0.0 > logs/best_mechanism_a_1.20_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.30 --beta 0.0 > logs/best_mechanism_a_1.20_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.35 --beta 0.0 > logs/best_mechanism_a_1.35_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.40 --beta 0.0 > logs/best_mechanism_a_1.40_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.45 --beta 0.0 > logs/best_mechanism_a_1.45_b_0.0.log 2>&1 &
# nohup python 03_best_mechanism.py -f ./data/synthetic/indian_farmers_carbon_market_250.csv -n 100 --alpha 1.50 --beta 0.0 > logs/best_mechanism_a_1.50_b_0.0.log 2>&1 &

