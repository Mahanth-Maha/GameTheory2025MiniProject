import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functools 
import math
from itertools import combinations, permutations
import argparse 


from utils import *
from metrics import *

# Overriding the functions(dynamic data param added)
def shapley_value_exact(all_farmer_ids, v_func, **v_func_params):
    n = len(all_farmer_ids)
    if n == 0: 
        return {}
    shapley_values = {f_id: 0.0 for f_id in all_farmer_ids}
    n_factorial = math.factorial(n)

    for p in permutations(all_farmer_ids):
        current_coalition_ids = []
        v_prev = 0.0
        for farmer_id in p:
            current_coalition_ids.append(farmer_id)
            v_current = v_func(current_coalition_ids, **v_func_params)
            marginal_contribution = v_current - v_prev
            shapley_values[farmer_id] += marginal_contribution
            v_prev = v_current

    for f_id in shapley_values:
        shapley_values[f_id] /= n_factorial
    return shapley_values

def shapley_value_monte_carlo(all_farmer_ids, v_func, n_samples=1000, **v_func_params):
    n = len(all_farmer_ids)
    if n == 0: return {}
    shapley_values = {f_id: 0.0 for f_id in all_farmer_ids}
    farmer_indices = list(range(n))

    for _ in range(n_samples):
        p_indices = np.random.permutation(farmer_indices)
        current_coalition_ids = []
        v_prev = 0.0
        for idx in p_indices:
            farmer_id = all_farmer_ids[idx]
            current_coalition_ids.append(farmer_id)
            v_current = v_func(current_coalition_ids, **v_func_params)
            marginal_contribution = v_current - v_prev
            shapley_values[farmer_id] += marginal_contribution
            v_prev = v_current

    for f_id in shapley_values:
        shapley_values[f_id] /= n_samples
    return shapley_values

def get_characteristic_function_values(all_farmer_ids, v_func, **v_func_params):
    char_values = {}
    n = len(all_farmer_ids)
    grand_coalition_key = tuple(sorted(all_farmer_ids))
    char_values[grand_coalition_key] = v_func(list(grand_coalition_key), **v_func_params)

    for r in range(1, n):
        for subset_ids_tuple in combinations(all_farmer_ids, r):
            sorted_subset_ids = tuple(sorted(subset_ids_tuple))
            char_values[sorted_subset_ids] = v_func(list(sorted_subset_ids), **v_func_params)
    return char_values


parser = argparse.ArgumentParser(description="Run Aggregator Model VCM Simulations.")
parser.add_argument('--aggregator_model', type=str, required=True,choices=list(AGGREGATOR_MODELS.keys()),help='Name of the aggregator payment model to use.')
parser.add_argument('--n_farmers', type=int, default=15,help='Number of farmers to sample (<=15 for Core check).')
parser.add_argument('--data_file', type=str,default='./data/synthetic/indian_farmers_carbon_market_250.csv',help='Path to the base farmer dataset CSV.')
parser.add_argument('--output_dir_base', type=str, default='./logs',help='Base directory for saving logs, results, and plots.')
parser.add_argument('--alpha', type=float, default=1.25,help='Base potential parameter alpha.')
parser.add_argument('--beta', type=float, default=0.0,help='Base potential parameter beta.')
parser.add_argument('--c_fixed', type=float, default=10000,help='Aggregator fixed cost.')
parser.add_argument('--c_var', type=float, default=300,help='Aggregator variable cost per farmer.')
parser.add_argument('--delta_max', type=float, default=0.5,help='Maximum primary commission rate (delta) to test.')
parser.add_argument('--delta_steps', type=int, default=11,help='Number of steps for delta range (e.g., 11 for 0 to max in 0.05 steps).')

parser.add_argument('--delta2', type=float, default=0.1,help='Secondary commission rate (for two_tier_commission model).')
parser.add_argument('--eta', type=float, default=1.1,help='Target multiplier eta > 1 (for target_commission model).')
parser.add_argument('--fee_per_farmer', type=float, default=50,help='Fixed fee per farmer (for fixed_fee_plus_surplus model).')
parser.add_argument('--seed', type=int, default=24004,help='Random seed for sampling.')
parser.add_argument('--shapley_samples', type=int, default=1000,help='Number of Monte Carlo samples per farmer for Shapley value.')
parser.add_argument('--skip_core', action='store_true',help='Skip Core check calculations (recommended for N > 15).')
parser.add_argument('--plot_dir', type=str, default=None,help='Directory for saving plots (overrides output_dir_base).')

args = parser.parse_args()


N_FARMERS = args.n_farmers
SEED = args.seed
DATA_FILE = args.data_file
MODEL_NAME = args.aggregator_model


AGG_DIR = os.path.join(args.output_dir_base, 'aggregator_model', MODEL_NAME)
if args.plot_dir:
    PLOT_DIR = args.plot_dir
else:
    PLOT_DIR = os.path.join(args.output_dir_base, 'plots', 'aggregator_model', MODEL_NAME )
os.makedirs(AGG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ALPHA = args.alpha
BETA = args.beta
C_FIXED = args.c_fixed
C_VAR = args.c_var

DELTA_VALUES = np.linspace(0.0, args.delta_max, args.delta_steps)
DELTA2 = args.delta2
ETA = args.eta
FEE_PER_FARMER = args.fee_per_farmer

EXACT_SHAPLEY_THRESHOLD = 10
SHAPLEY_SAMPLES = args.shapley_samples if N_FARMERS > EXACT_SHAPLEY_THRESHOLD else None

CORE_CHECK_THRESHOLD = 15
PERFORM_CORE_CHECK = (N_FARMERS <= CORE_CHECK_THRESHOLD) and (not args.skip_core)

if __name__ == "__main__":
    full_data = pd.read_csv(DATA_FILE)
    if N_FARMERS < len(full_data):
        data = full_data.sample(n=N_FARMERS, random_state=SEED).reset_index(drop=True)
        print(f"[+] Sampled {N_FARMERS} farmers from {DATA_FILE}")
    else:
        N_FARMERS = len(full_data)
        data = full_data.reset_index(drop=True)
        print(f"[+] Using all {N_FARMERS} farmers from {DATA_FILE}")
    farmer_ids = data['Farmer_ID'].tolist()
    standalone_payoffs_dict = data.set_index('Farmer_ID')['Standalone_Payoff_INR'].to_dict()
    sum_standalone_payoffs_R_N = data['Standalone_Payoff_INR'].sum()
    
    results_list = []
    print(f"\n[+] Running Aggregator Model Simulation: '{MODEL_NAME}' for N={N_FARMERS}")
    print(f"[>] Base Params: alpha={ALPHA}, beta={BETA}")
    print(f"[>] Agg Cost Params: C_fixed={C_FIXED}, C_var={C_VAR}")
    print(f"[>] Primary Commission (delta) Range: {DELTA_VALUES}")
    if MODEL_NAME == 'two_tier_commission': print(f"[>] Secondary Commission (delta2): {DELTA2}")
    if MODEL_NAME == 'target_commission': print(f"[>] Target Multiplier (eta): {ETA}")
    if MODEL_NAME == 'fixed_fee_plus_surplus': print(f"[>] Fixed Fee Per Farmer: {FEE_PER_FARMER}")
    if PERFORM_CORE_CHECK: print("[>] Core checks will be performed.")
    else: print("[>] Core checks will be skipped.")


    start_time_total = time.time()

    all_char_values_vf_by_delta = {}
    if PERFORM_CORE_CHECK:
        print("[+] Precomputing characteristic values for Core checks...")
        for delta_val in DELTA_VALUES:
             v_func_for_core = get_vf_func_for_model(data, MODEL_NAME, ALPHA, BETA, C_FIXED, C_VAR,delta_val, DELTA2, ETA, FEE_PER_FARMER)
             all_char_values_vf_by_delta[delta_val] = get_characteristic_function_values(
                 farmer_ids, v_func=v_func_for_core
             )
        print("[+] Precomputation done.")

    for delta_val in DELTA_VALUES:
        print(f"  -> Testing delta = {delta_val:.3f}...")
        start_time_delta = time.time()

        v_func_for_shapley = get_vf_func_for_model(data,MODEL_NAME,ALPHA,BETA,C_FIXED,C_VAR,delta_val,DELTA2,ETA,FEE_PER_FARMER )

        if N_FARMERS <= EXACT_SHAPLEY_THRESHOLD:
            farmer_payoffs_xf = shapley_value_exact(farmer_ids, v_func=v_func_for_shapley)
        else:
            farmer_payoffs_xf = shapley_value_monte_carlo(farmer_ids, v_func=v_func_for_shapley, n_samples=SHAPLEY_SAMPLES)

        payoffs_list = [farmer_payoffs_xf.get(fid, 0) for fid in farmer_ids]
        avg_farmer_payoff = np.mean(payoffs_list) if payoffs_list else 0
        gini = gini_coefficient(payoffs_list)

        gains_abs = []
        gains_pct = []
        ir_met_count = 0
        for farmer_id in farmer_ids:
            r_i = standalone_payoffs_dict.get(farmer_id, 0)
            x_i = farmer_payoffs_xf.get(farmer_id, 0)
            gain = x_i - r_i
            gains_abs.append(gain)
            gains_pct.append((gain / r_i * 100) if r_i > 0 else 0)
            if x_i >= r_i - 1e-6:
                ir_met_count += 1
        avg_abs_gain = np.mean(gains_abs) if gains_abs else 0
        avg_pct_gain = np.mean(gains_pct) if gains_pct else 0
        ir_met_percentage = (ir_met_count / N_FARMERS * 100) if N_FARMERS > 0 else 100

        aggregator_profit_piA, total_farmer_value_vfN = calculate_agg_profit_and_vf(farmer_ids, data, MODEL_NAME, ALPHA, BETA, C_FIXED, C_VAR,delta_val, DELTA2, ETA, FEE_PER_FARMER)
        
        V_potential_N = characteristic_function_v(farmer_ids, data, alpha=ALPHA, beta=BETA)
        C_A_N = C_FIXED + C_VAR * N_FARMERS if N_FARMERS > 0 else 0
        V_net_available_N = max(0.0, V_potential_N - C_A_N)
        surplus_N = max(0.0, V_net_available_N - sum_standalone_payoffs_R_N)

        core_stable = None
        if PERFORM_CORE_CHECK:
            print(f"[>] Checking Core stability..." , end= '')
            char_values_vf = all_char_values_vf_by_delta.get(delta_val, None)
            if char_values_vf:
                core_stable, _ = is_in_core(farmer_payoffs_xf, farmer_ids, char_values_vf)
                if core_stable == True:
                    print(f"\tStable")
                elif core_stable == False:
                    print("\tUnstable")
            else:
                print("\n\n[!] Precomputed characteristic values not found for core check\n\n")
                core_stable = "Error"

        results_list.append({
            'delta': delta_val,
            'alpha': ALPHA, 
            'beta': BETA, 
            'C_fixed': C_FIXED,
            'C_var': C_VAR,
            'avg_farmer_payoff': avg_farmer_payoff,
            'avg_abs_gain': avg_abs_gain,
            'avg_pct_gain': avg_pct_gain,
            'gini_coefficient': gini,
            'ir_met_percentage': ir_met_percentage,
            'aggregator_profit_piA': aggregator_profit_piA,
            'total_farmer_value_vfN': total_farmer_value_vfN,
            'V_potential_N': V_potential_N,
            'C_A_N': C_A_N,
            'V_net_available_N': V_net_available_N,
            'Surplus_N': surplus_N,
            'R_N': sum_standalone_payoffs_R_N,
            'is_in_core': core_stable if PERFORM_CORE_CHECK else 'Not Checked'
        })
        print(f"[>]  Done in {time.time() - start_time_delta:.2f}s")

    results_df = pd.DataFrame(results_list)
    print(f"\n[+] Simulation Results Summary ({MODEL_NAME}):")
    display_cols = ['delta', 'avg_farmer_payoff', 'avg_abs_gain', 'ir_met_percentage',
                    'aggregator_profit_piA', 'Surplus_N', 'gini_coefficient', 'is_in_core']
    print(results_df[display_cols].round(2))

    csv_filename = os.path.join(AGG_DIR, f'aggregator_{MODEL_NAME}_results_n{N_FARMERS}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"\n[+] Results saved to {csv_filename}")

    def plotit():
        print(f"\n[+] Generating plots in {PLOT_DIR}...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)
        fig.suptitle(f'Impact of Aggregator Commission ($\\delta$) on VCM Outcomes (N={N_FARMERS})', fontsize=14)

        x_axis_var = 'delta'
        x_label = 'Aggregator Commission Rate ($\\delta$)'


        axes[0, 0].plot(results_df[x_axis_var], results_df['avg_farmer_payoff'], marker='o')
        axes[0, 0].set_title('Avg. Farmer Payoff (Shapley on $v_F$)')
        axes[0, 0].set_ylabel('Avg Payoff (INR)')
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

        axes[0, 1].plot(results_df[x_axis_var], results_df['avg_abs_gain'], marker='o', color='orange')
        axes[0, 1].set_title('Avg. Farmer Gain ($x_i - r_i$)')
        axes[0, 1].set_ylabel('Avg Gain (INR)')
        axes[0, 1].axhline(0, color='red', linestyle=':', linewidth=1, label='Break-even (Gain=0)')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

        axes[1, 0].plot(results_df[x_axis_var], results_df['ir_met_percentage'], marker='o', color='green')
        axes[1, 0].set_title('Farmer Participation (IR Met %)')
        axes[1, 0].set_ylabel('% Farmers with ($x_i ≥ r_i$)')
        axes[1, 0].set_ylim(-5, 105)
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

        axes[1, 1].plot(results_df[x_axis_var], results_df['aggregator_profit_piA'], marker='o', color='purple')
        axes[1, 1].set_title('Aggregator Profit ($\\pi_A(N)$)')
        axes[1, 1].set_ylabel('Profit (INR)')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

        axes[2, 0].plot(results_df[x_axis_var], results_df['gini_coefficient'], marker='o', color='brown')
        axes[2, 0].set_title('Fairness (Gini Coefficient of Farmer Payoffs)')
        axes[2, 0].set_ylabel('Gini Coefficient')
        axes[2, 0].grid(True, linestyle='--', alpha=0.6)

        axes[2, 0].set_xlabel(x_label)
        
        if PERFORM_CORE_CHECK and 'is_in_core' in results_df.columns and not results_df['is_in_core'].isin(['Not Checked', 'Error']).all():
            core_numeric = results_df['is_in_core'].map({True: 1, False: 0, None: np.nan, 'Error': np.nan, 'Not Checked': np.nan})
            axes[2, 1].plot(results_df[x_axis_var], core_numeric, marker='o', color='cyan', linestyle='--')
            axes[2, 1].set_title('Core Stability (1=Stable, 0=Unstable)')
            axes[2, 1].set_yticks([0, 1])
            axes[2, 1].set_yticklabels(['Unstable', 'Stable'])
            axes[2, 1].set_ylim(-0.5, 1.5)
        else:
            axes[2, 1].text(0.25, 0.5, 'Core Check Not Performed\nResults Invalid',ha='center', va='center', fontsize=10, color='gray', wrap=True)
            axes[2, 1].set_title('Core Stability')
            axes[2, 1].set_yticks([])

        axes[2, 1].set_xlabel(x_label)
        axes[2, 1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plot_filename = os.path.join(PLOT_DIR, f'aggregator_impact_{MODEL_NAME}_n{N_FARMERS}.png')
        plt.savefig(plot_filename)
        print(f"[+] Main impact plot saved to {plot_filename}")
        plt.close()

    plotit()
    print(f"\n[+] Analysis for model '{MODEL_NAME}' complete. Total time: {time.time() - start_time_total:.2f}s")




# nohup python  06_agg_models.py --aggregator_model commission_net --n_farmers 5 > logs/aggregator_model/commission_net/results_n5.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model commission_net --n_farmers 15 > logs/aggregator_model/commission_net/results_n15.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model commission_net --n_farmers 250 > logs/aggregator_model/commission_net/results_n250.log 2>&1 &


# nohup python  06_agg_models.py --aggregator_model commission_surplus --n_farmers 5 > logs/aggregator_model/commission_surplus/results_n5.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model commission_surplus --n_farmers 15 > logs/aggregator_model/commission_surplus/results_n15.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model commission_surplus --n_farmers 250 --skip_core > logs/aggregator_model/commission_surplus/results_n250.log 2>&1 &



# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 5 --eta 1.1  --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_1/ > logs/aggregator_model/target_commission/results_eta_1_1_n5.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 15 --eta 1.1 --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_1/ > logs/aggregator_model/target_commission/results_eta_1_1_n15.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 250 --eta 1.1 --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_1/ > logs/aggregator_model/target_commission/results_eta_1_1_n250.log 2>&1 &

# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 5 --eta 1.2 --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_2/ > logs/aggregator_model/target_commission/results_eta_1_2_n5.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 15 --eta 1.2 --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_2/ > logs/aggregator_model/target_commission/results_eta_1_2_n15.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model target_commission --n_farmers 250 --eta 1.2 --plot_dir ./logs/plots/aggregator_model/target_commission/eta_1_2/ > logs/aggregator_model/target_commission/results_eta_1_2_n250.log 2>&1 &




# nohup python  06_agg_models.py --aggregator_model two_tier_commission --n_farmers 5 --delta_max 0.3 --delta2 0.05 > logs/aggregator_model/two_tier_commission/results_n5_delta2_0_05.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model two_tier_commission --n_farmers 15 --delta_max 0.3 --delta2 0.05 > logs/aggregator_model/two_tier_commission/results_n15_delta2_0_05.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model two_tier_commission --n_farmers 250 --delta_max 0.3 --delta2 0.05 > logs/aggregator_model/two_tier_commission/results_n250_delta2_0_05.log 2>&1 &


# nohup python  06_agg_models.py --aggregator_model fixed_fee_plus_surplus --n_farmers 5 --skip_core --fee_per_farmer 100 > logs/aggregator_model/fixed_fee_plus_surplus/results_n5_fee_100.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model fixed_fee_plus_surplus --n_farmers 15 --skip_core --fee_per_farmer 100 > logs/aggregator_model/fixed_fee_plus_surplus/results_n15_fee_100.log 2>&1 &
# nohup python  06_agg_models.py --aggregator_model fixed_fee_plus_surplus --n_farmers 250 --skip_core --fee_per_farmer 100 > logs/aggregator_model/fixed_fee_plus_surplus/results_n250_fee_100.log 2>&1 &


# nohup python 06_agg_models.py --aggregator_model guranteeded_fee_for_all --delta_max 1.0 --delta_steps 21 --n_farmers 5 > logs/aggregator_model/guranteeded_fee_for_all/results_n5.log 2>&1 &
# nohup python 06_agg_models.py --aggregator_model guranteeded_fee_for_all --delta_max 1.0 --delta_steps 21 --n_farmers 15 > logs/aggregator_model/guranteeded_fee_for_all/results_n15.log 2>&1 &
# nohup python 06_agg_models.py --aggregator_model guranteeded_fee_for_all --delta_max 1.0 --delta_steps 21 --n_farmers 250 > logs/aggregator_model/guranteeded_fee_for_all/results_n250.log 2>&1 &
