import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import functools 
import math
from itertools import combinations, permutations


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

def calculate_vf_aggregator(coalition_ids, data, alpha0, beta0, C_fixed, C_var, delta_commission):
    if not coalition_ids:
        return 0.0
    if not isinstance(coalition_ids, list):
        coalition_ids = list(coalition_ids)
    V_potential = characteristic_function_v(coalition_ids, data, alpha=alpha0, beta=beta0)
    C_A = C_fixed + C_var * len(coalition_ids) if coalition_ids else 0.0
    V_net_available = max(0.0, V_potential - C_A)

    v_F = (1.0 - delta_commission) * V_net_available
    return v_F

# N_FARMERS = 10
N_FARMERS = 250
SEED = 24004
DATA_FILE = './data/synthetic/indian_farmers_carbon_market_250.csv'
PLOT_DIR = './logs/plots/aggregator_model/'
os.makedirs(PLOT_DIR, exist_ok=True)
AGG_DIR = './logs/aggregator_model/'
os.makedirs(AGG_DIR, exist_ok=True)

ALPHA0 = 1.25 
BETA0 = 0.0

C_FIXED = 10000 
C_VAR = 300

DELTA_VALUES = np.linspace(0.0, 0.50, 11)

EXACT_SHAPLEY_THRESHOLD = 10
SHAPLEY_SAMPLES = 1000 

CORE_CHECK_THRESHOLD = 15
PERFORM_CORE_CHECK = (N_FARMERS <= CORE_CHECK_THRESHOLD)

if __name__ == "__main__":
    # pass

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


    results_list = []
    print(f"\n[+] Running Aggregator Model Simulation for N={N_FARMERS}")
    print(f"[>] Base Params: alpha0={ALPHA0}, beta0={BETA0}")
    print(f"[>] Agg Cost Params: C_fixed={C_FIXED}, C_var={C_VAR}")
    print(f"[>] Commission Rates (DELTA): {DELTA_VALUES}")

    start_time_total = time.time()
    all_char_values_vf_by_delta = {}
    if PERFORM_CORE_CHECK:
        print("[+] Precomputing characteristic values for Core checks...")
        for delta_commission in DELTA_VALUES:
            v_func_current_delta = functools.partial(
                calculate_vf_aggregator,
                data=data,
                alpha0=ALPHA0,
                beta0=BETA0,
                C_fixed=C_FIXED,
                C_var=C_VAR,
                delta_commission=delta_commission
            )
            all_char_values_vf_by_delta[delta_commission] = get_characteristic_function_values(
                farmer_ids, v_func=v_func_current_delta
            )
        print("[+] Precomputation done.")

    for delta_commission in DELTA_VALUES:
        print(f"  -> Testing delta = {delta_commission:.3f}...")
        start_time_delta = time.time()

        v_func_current_delta = functools.partial(
            calculate_vf_aggregator,
            data=data,
            alpha0=ALPHA0,
            beta0=BETA0,
            C_fixed=C_FIXED,
            C_var=C_VAR,
            delta_commission=delta_commission
        )

        if N_FARMERS <= EXACT_SHAPLEY_THRESHOLD:
            farmer_payoffs_xf = shapley_value_exact(farmer_ids, v_func=v_func_current_delta)
        else:
            farmer_payoffs_xf = shapley_value_monte_carlo(farmer_ids, v_func=v_func_current_delta, n_samples=SHAPLEY_SAMPLES)

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

        V_potential_N = characteristic_function_v(farmer_ids, data, alpha=ALPHA0, beta=BETA0)
        C_A_N = C_FIXED + C_VAR * N_FARMERS if N_FARMERS > 0 else 0
        V_net_available_N = max(0.0, V_potential_N - C_A_N)
        aggregator_profit_piA = delta_commission * V_net_available_N
        total_farmer_value_vfN = (1.0 - delta_commission) * V_net_available_N

        core_stable = None
        if PERFORM_CORE_CHECK:
            print(f"[>] Checking Core stability..." , end= '')
            char_values_vf = all_char_values_vf_by_delta.get(delta_commission, None)
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
            'delta_commission': delta_commission,
            'alpha0': ALPHA0,
            'beta0': BETA0,
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
            'is_in_core': core_stable if PERFORM_CORE_CHECK else 'Not Checked'
        })
        print(f"[>] {delta_commission} --> Done in {time.time() - start_time_delta:.2f}s")

    results_df = pd.DataFrame(results_list)
    print("\n[+] Simulation Results Summary:")
    display_cols = ['delta_commission', 'avg_farmer_payoff', 'avg_abs_gain','ir_met_percentage', 'aggregator_profit_piA', 'gini_coefficient', 'is_in_core']
    print(results_df[display_cols].round(2))

    csv_filename = os.path.join(AGG_DIR, f'aggregator_model_results_n{N_FARMERS}_delta.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"\n[+] Results saved to {csv_filename}")
    
    def plotit():
        print(f"\n[+] Generating plots in {PLOT_DIR}...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True)
        fig.suptitle(f'Impact of Aggregator Commission ($\\delta$) on VCM Outcomes (N={N_FARMERS})', fontsize=14)

        x_axis_var = 'delta_commission'
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
        plot_filename = os.path.join(PLOT_DIR, f'aggregator_impact_vs_delta_n{N_FARMERS}.png')
        plt.savefig(plot_filename)
        print(f"[+] Main impact plot saved to {plot_filename}")
        plt.close()

    plotit()
    print(f"\n[+] Done in Total time: {time.time() - start_time_total:.2f}s")
    
# nohup python3 05_Aggregator_as_a_player.py > logs/aggregator_model/aggregator_model_v1.log 2>&1 &
# nohup python3 05_Aggregator_as_a_player.py > logs/aggregator_model/aggregator_model_v2_n_250.log 2>&1 &