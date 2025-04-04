from utils import *

def run_vcg_auction_mechanism(data, price_per_credit):
    if data.empty:
        return {'payoffs': {}, 'buyer_cost': 0.0, 'winners': [], 'total_surplus': 0.0}

    bids = data.set_index('Farmer_ID')['True_Cost_per_Credit_INR'].to_dict()
    credits = data.set_index('Farmer_ID')['Potential_Carbon_Credits_tCO2e'].to_dict()

    potential_winners_data = data[data['True_Cost_per_Credit_INR'] <= price_per_credit].copy()
    winners = potential_winners_data['Farmer_ID'].tolist()

    if not winners:
        return {'payoffs': {}, 'buyer_cost': 0.0, 'winners': [], 'total_surplus': 0.0, 'mechanism_name': 'VCG Auction'}

    excluded_farmers = data[data['True_Cost_per_Credit_INR'] > price_per_credit]
    if not excluded_farmers.empty:
        threshold_cost_per_credit = excluded_farmers['True_Cost_per_Credit_INR'].min()
    else:
        threshold_cost_per_credit = price_per_credit

    payoffs = {}
    total_buyer_cost = 0.0
    total_true_cost_winners = 0.0

    for farmer_id in winners:
        farmer_credits = credits[farmer_id]
        payment = threshold_cost_per_credit * farmer_credits
        payoffs[farmer_id] = round(payment, 2)
        total_buyer_cost += payment
        total_true_cost_winners += bids[farmer_id] * farmer_credits

    all_farmers = data['Farmer_ID'].tolist()
    for f_id in all_farmers:
        if f_id not in payoffs:
            payoffs[f_id] = 0.0

    total_buyer_value = price_per_credit * potential_winners_data['Potential_Carbon_Credits_tCO2e'].sum()
    total_surplus = total_buyer_value - total_true_cost_winners

    return {
        'payoffs': payoffs,
        'buyer_cost': round(total_buyer_cost, 2),
        'winners': winners,
        'total_surplus': round(total_surplus, 2),
        'mechanism_name': 'VCG Auction'
    }


def run_uniform_price_auction_mechanism(data, buyer_total_credits_demand):
    if data.empty or buyer_total_credits_demand <= 0:
        return {'payoffs': {}, 'buyer_cost': 0.0, 'winners': [], 'clearing_price': 0.0 , 'mechanism_name': 'Uniform Price Auction'}

    data_sorted = data.sort_values('True_Cost_per_Credit_INR').copy()
    data_sorted['Cumulative_Credits'] = data_sorted['Potential_Carbon_Credits_tCO2e'].cumsum()

    winning_data = data_sorted[data_sorted['Cumulative_Credits'] <= buyer_total_credits_demand]

    if len(winning_data) < len(data_sorted):
        first_excluded_idx = len(winning_data)
        clearing_price = data_sorted.iloc[first_excluded_idx]['True_Cost_per_Credit_INR']
        if not winning_data.empty and winning_data.iloc[-1]['Cumulative_Credits'] == buyer_total_credits_demand:
             clearing_price = winning_data.iloc[-1]['True_Cost_per_Credit_INR']

    elif not winning_data.empty:
         clearing_price = winning_data.iloc[-1]['True_Cost_per_Credit_INR']
    else:
         clearing_price = 0.0
         winners = []
         payoffs = {f_id: 0.0 for f_id in data['Farmer_ID']}
         total_buyer_cost = 0.0

    if clearing_price > 0 and not winning_data.empty:
        winners = winning_data['Farmer_ID'].tolist()
        payoffs = {}
        total_buyer_cost = 0.0
        credits = data.set_index('Farmer_ID')['Potential_Carbon_Credits_tCO2e'].to_dict()

        for farmer_id in winners:
            payment = clearing_price * credits[farmer_id]
            payoffs[farmer_id] = round(payment, 2)
            total_buyer_cost += payment

        all_farmers = data['Farmer_ID'].tolist()
        for f_id in all_farmers:
            if f_id not in payoffs:
                payoffs[f_id] = 0.0
    elif not winning_data.empty :
         winners = winning_data['Farmer_ID'].tolist()
         payoffs = {f_id: 0.0 for f_id in data['Farmer_ID']}
         total_buyer_cost = 0.0


    return {
        'payoffs': payoffs,
        'buyer_cost': round(total_buyer_cost, 2),
        'winners': winners,
        'clearing_price': clearing_price,
        'mechanism_name': 'Uniform Price Auction'
    }


def run_shapley_allocation_mechanism(data, market_price_per_credit, alpha, beta, use_exact_shapley, shapley_samples = 1000, core_check_threshold = 15):
    farmer_ids = data['Farmer_ID'].tolist()
    n = len(farmer_ids)
    if n == 0:
        return {'payoffs': {}, 'is_in_core': None, 'blocking_coalitions': None , 'mechanism_name': 'Shapley Allocation'}

    gt_params = {'alpha': alpha, 'beta': beta}

    if use_exact_shapley and n <= 10:
        payoffs = shapley_value_exact(farmer_ids, data, characteristic_function_v, gt_params)
    else:
        payoffs = shapley_value_monte_carlo(farmer_ids, data, characteristic_function_v, gt_params, n_samples=shapley_samples)

    is_in_core_result = None
    blocking_coalitions = None
    core_check_performed = False
    if n <= core_check_threshold:
        try:
            char_values = get_characteristic_function_values(farmer_ids, data, characteristic_function_v, gt_params)
            is_in_core_result, blocking_coalitions = is_in_core(payoffs, farmer_ids, char_values)
            core_check_performed = True
        except Exception as e:
            print(f"[-] Core check failed for Shapley Mechanism (N={n}): {e}")
            is_in_core_result = None
            blocking_coalitions = None


    return {
        'payoffs': payoffs,
        'is_in_core': is_in_core_result,
        'blocking_coalitions': blocking_coalitions,
        'core_check_performed': core_check_performed,
        'mechanism_name': 'Shapley Allocation'
    }