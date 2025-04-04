import numpy as np
from itertools import combinations, permutations
import math

# Utility functions

def linear_utility(x, params):
    return x

def log_utility(x, params):
    a = params.get('a', 1.0)
    b = params.get('b', 0.0)
    if x <= 0:
        return -1e9
    return a * math.log(x) + b

def power_utility(x, params):
    eta = params.get('eta', 1.0)
    if x <= 0:
         return -1e9
    if np.isclose(eta, 1.0):
        return math.log(x)
    else:
        return (x**(1 - eta)) / (1 - eta)

UTILITY_FUNCTIONS = {
    'linear': linear_utility,
    'logarithmic': log_utility,
    'power': power_utility
}



# Coalitiona Value Function

def characteristic_function_v(farmer_ids,data,alpha = 1.0, beta = 0.0,  gamma = 0.5):
    if not farmer_ids:
        return 0.0

    coalition_data = data[data['Farmer_ID'].isin(farmer_ids)]
    if coalition_data.empty:
        return 0.0

    sum_standalone_payoffs = coalition_data['Standalone_Payoff_INR'].sum()
    value = alpha * sum_standalone_payoffs + beta * (sum_standalone_payoffs ** 2)

    return value

def get_characteristic_function_values( all_farmer_ids, data, v_func = characteristic_function_v, v_func_params = {}):
    char_values = {}
    n = len(all_farmer_ids)
    for r in range(1, n + 1):
        for subset_ids_tuple in combinations(all_farmer_ids, r):
            sorted_subset_ids = tuple(sorted(subset_ids_tuple))
            subset_ids_list = list(sorted_subset_ids)
            char_values[sorted_subset_ids] = v_func(subset_ids_list, data, **v_func_params)
    return char_values



# Shapley Value

def shapley_value_exact( all_farmer_ids, data, v_func = characteristic_function_v, v_func_params = {}):
    n = len(all_farmer_ids)
    if n == 0: 
        return {}
    if n > 10:
        print(f"[-] Number of farmers ({n}) exceeds 10. Using Monte Carlo")
        return shapley_value_monte_carlo(all_farmer_ids, data, v_func, v_func_params, n_samples=1000*n)

    shapley_values = {f_id: 0.0 for f_id in all_farmer_ids}
    n_factorial = math.factorial(n)

    for p in permutations(all_farmer_ids):
        current_coalition_ids = []
        v_prev = 0.0
        for farmer_id in p:
            current_coalition_ids.append(farmer_id)
            v_current = v_func(current_coalition_ids, data, **v_func_params)
            marginal_contribution = v_current - v_prev
            shapley_values[farmer_id] += marginal_contribution
            v_prev = v_current

    for f_id in shapley_values:
        shapley_values[f_id] /= n_factorial

    return shapley_values

def shapley_value_monte_carlo( all_farmer_ids, data, v_func = characteristic_function_v, v_func_params = {}, n_samples = 1000):
    n = len(all_farmer_ids)
    if n == 0: 
        return {}
    if not all_farmer_ids: 
        return {}

    shapley_values = {f_id: 0.0 for f_id in all_farmer_ids}
    farmer_indices = list(range(n))

    for _ in range(n_samples):
        p_indices = np.random.permutation(farmer_indices)
        current_coalition_ids = []
        v_prev = 0.0
        for idx in p_indices:
            farmer_id = all_farmer_ids[idx]
            current_coalition_ids.append(farmer_id)
            v_current = v_func(current_coalition_ids, data, **v_func_params)
            marginal_contribution = v_current - v_prev
            shapley_values[farmer_id] += marginal_contribution
            v_prev = v_current

    for f_id in shapley_values:
        shapley_values[f_id] /= n_samples

    return shapley_values


# Core 

def is_in_core( payoff_vector, all_farmer_ids, char_values, tolerance = 1e-6):
    n = len(all_farmer_ids)
    if not all_farmer_ids or not payoff_vector: 
        return (False, None)

    total_payoff = sum(payoff_vector.values())
    grand_coalition_key = tuple(sorted(all_farmer_ids))
    if grand_coalition_key not in char_values:
         print(f"[-] Grand coalition value not found in char_values.")
         return False, None
    grand_coalition_value = char_values[grand_coalition_key]

    if not np.isclose(total_payoff, grand_coalition_value, atol=tolerance):
        print(f"[-] Total payoff {total_payoff:.2f} != Grand Coalition value {grand_coalition_value:.2f}")
        return False, []

    blocking_coalitions = []
    for subset_ids_tuple in char_values:
        if len(subset_ids_tuple) == n:
            continue

        subset_payoff_sum = sum(payoff_vector.get(f_id, 0) for f_id in subset_ids_tuple)
        subset_value = char_values[subset_ids_tuple]

        if subset_value > subset_payoff_sum + tolerance:
             blocking_coalitions.append(subset_ids_tuple)
             # print(f"[-] Blocking Coalition Found: {subset_ids_tuple}. Value={subset_value:.2f}, Allocated={subset_payoff_sum:.2f}")


    if not blocking_coalitions:
        return True, None
    else:
        return False, blocking_coalitions 



# VCG Auction

def run_vcg_auction(data,price_per_credit):
    if data.empty:
        return [], {}, 0.0, 0.0

    potential_winners = data[data['True_Cost_per_Credit_INR'] <= price_per_credit].copy()

    if potential_winners.empty:
        return [], {}, 0.0, 0.0

    potential_winners.sort_values('True_Cost_per_Credit_INR', inplace=True)

    excluded_farmers = data[data['True_Cost_per_Credit_INR'] > price_per_credit]
    if not excluded_farmers.empty:
        threshold_cost = excluded_farmers['True_Cost_per_Credit_INR'].min()
    else:
        threshold_cost = price_per_credit

    winners = potential_winners['Farmer_ID'].tolist()
    payments = {}
    total_payments = 0.0
    total_true_cost_winners = 0.0

    for farmer_id in winners:
        farmer_data = potential_winners[potential_winners['Farmer_ID'] == farmer_id].iloc[0]
        num_credits = farmer_data['Potential_Carbon_Credits_tCO2e']
        payment = max(threshold_cost * num_credits, farmer_data['True_Cost_per_Credit_INR'] * num_credits)
        payments[farmer_id] = round(payment, 2)
        total_payments += payment
        total_true_cost_winners += farmer_data['True_Cost_per_Credit_INR'] * num_credits

    total_buyer_value = price_per_credit * potential_winners['Potential_Carbon_Credits_tCO2e'].sum()
    total_surplus = total_buyer_value - total_true_cost_winners

    return winners, payments, round(total_surplus, 2), round(total_payments, 2)