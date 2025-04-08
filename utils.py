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


# Aggregator Models 

# Aggregator takes delta commission on NET value after costs
def agg_model_commission_net(V_potential, C_A, R_S, delta, **kwargs):
    V_net_available = max(0.0, V_potential - C_A)
    aggregator_profit = delta * V_net_available
    net_value_for_farmers = V_net_available - aggregator_profit
    return aggregator_profit, net_value_for_farmers

# Aggregator takes delta commission on SURPLUS above baseline sum R_S
def agg_model_commission_surplus(V_potential, C_A, R_S, delta, **kwargs):
    V_net_available = max(0.0, V_potential - C_A)
    surplus = max(0.0, V_net_available - R_S)
    aggregator_profit = delta * surplus
    net_value_for_farmers = V_net_available - aggregator_profit
    return aggregator_profit, net_value_for_farmers

# Aggregator gets delta commission ONLY on value EXCEEDING a target (eta * R_S)
def agg_model_target_commission(V_potential, C_A, R_S, delta, eta=1.1, **kwargs):
    V_net_available = max(0.0, V_potential - C_A)
    target_value = eta * R_S
    if V_net_available >= target_value:
        bonus_value = V_net_available - target_value
        aggregator_profit = delta * bonus_value
        net_value_for_farmers = V_net_available - aggregator_profit
    else:
        aggregator_profit = 0.0
        net_value_for_farmers = V_net_available
    return aggregator_profit, net_value_for_farmers

# Two commission rates - delta on value up to R_S, delta2 on surplus
def agg_model_two_tier_commission(V_potential, C_A, R_S, delta, delta2=0.1, **kwargs):
    V_net_available = max(0.0, V_potential - C_A)
    value_up_to_baseline = min(V_net_available, R_S)
    surplus_value = max(0.0, V_net_available - R_S)
    aggregator_profit = (delta * value_up_to_baseline) + (delta2 * surplus_value)
    net_value_for_farmers = V_net_available - aggregator_profit
    return aggregator_profit, net_value_for_farmers

# Fixed fee per farmer + delta commission on remaining surplus."""
def agg_model_fixed_fee_plus_surplus(V_potential, C_A, R_S, delta, fixed_fee_per_farmer=50, **kwargs):
    V_net_available = max(0.0, V_potential - C_A)
    num_farmers = kwargs.get('num_farmers', 0)
    aggregator_fixed_profit = fixed_fee_per_farmer * num_farmers

    surplus_for_commission = max(0.0, V_net_available - R_S - aggregator_fixed_profit)
    aggregator_commission_profit = delta * surplus_for_commission
    aggregator_profit = aggregator_fixed_profit + aggregator_commission_profit
    aggregator_profit = min(aggregator_profit, V_net_available)

    net_value_for_farmers = V_net_available - aggregator_profit
    return aggregator_profit, net_value_for_farmers


# Aggregator takes a fixed fee + delta commission on NET value after costs , farmers stanalone + surplus
def agg_model_guranteeded_fee_for_all(V_potential, C_A, R_S, delta, **kwargs):
    V_net = max(0.0, V_potential - R_S - C_A)
    aggregator_bonus = delta * V_net
    aggregator_total = C_A + aggregator_bonus
    farmer_surplus   = R_S + (1 - delta) * V_net
    return aggregator_total, farmer_surplus

AGGREGATOR_MODELS = {
    "commission_net": agg_model_commission_net,
    "commission_surplus": agg_model_commission_surplus,
    "target_commission": agg_model_target_commission,
    "two_tier_commission": agg_model_two_tier_commission,
    "fixed_fee_plus_surplus": agg_model_fixed_fee_plus_surplus,
    "guranteeded_fee_for_all": agg_model_guranteeded_fee_for_all, 
    "no_aggregator": lambda V_potential, C_A, R_S, **kwargs: (0.0, V_potential)
}

def calculate_agg_profit_and_vf(coalition_ids, data, model_name, alpha0, beta0, C_fixed, C_var, delta, delta2, eta, fee_per_farmer):
    if not coalition_ids:
        return 0.0, 0.0

    if not isinstance(coalition_ids, list):
        coalition_ids = list(coalition_ids)

    V_potential = characteristic_function_v(coalition_ids, data, alpha=alpha0, beta=beta0)
    C_A = C_fixed + C_var * len(coalition_ids) if coalition_ids else 0.0
    coalition_data = data[data['Farmer_ID'].isin(coalition_ids)]
    R_S = coalition_data['Standalone_Payoff_INR'].sum() if not coalition_data.empty else 0.0

    model_func = AGGREGATOR_MODELS.get(model_name)
    if not model_func:
        raise ValueError(f"Unknown aggregator model name: {model_name}")
        
    agg_profit, v_F = model_func(
        V_potential=V_potential,
        C_A=C_A,
        R_S=R_S,
        delta=delta,
        delta2=delta2,
        eta=eta,
        fixed_fee_per_farmer=fee_per_farmer,
        num_farmers=len(coalition_ids)
    )
    return agg_profit, v_F

def get_vf_func_for_model(data, model_name, alpha0, beta0, C_fixed, C_var, delta, delta2, eta, fee_per_farmer):
    def vf_calculator(coalition_ids):
        _, v_F = calculate_agg_profit_and_vf(coalition_ids, data, model_name, alpha0, beta0,C_fixed, C_var, delta, delta2, eta, fee_per_farmer)
        return v_F
    return vf_calculator