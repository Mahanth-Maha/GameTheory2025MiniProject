import numpy as np
import pandas as pd

def gini_coefficient(payoffs):
    if not payoffs or len(payoffs) == 0:
        return 0.0
    x = np.sort(np.array(payoffs, dtype=float))
    n = len(x)
    if n == 0 or np.sum(x) == 0:
        if n <= 1: 
            return 0.0
        if np.all(x==0): 
            return 0.0
        else: 
            return np.nan
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))
    return gini

def vcg_individual_rationality(payments, data) :
    if not payments: 
        return True

    for farmer_id, payment in payments.items():
        farmer_data = data[data['Farmer_ID'] == farmer_id]
        if farmer_data.empty:
            print(f"[!] Farmer {farmer_id} not found in data for IR check.")
            continue
        true_cost = farmer_data.iloc[0]['True_Cost_per_Credit_INR']
        num_credits = farmer_data.iloc[0]['Potential_Carbon_Credits_tCO2e']
        total_true_cost = true_cost * num_credits
        if payment < total_true_cost - 1e-6:
            print(f"[-] Farmer {farmer_id}, Payment={payment:.2f}, Cost={total_true_cost:.2f}")
            return False
    return True

def vcg_budget_balance(total_payments, total_buyer_value):
    return total_payments - total_buyer_value


def check_individual_rationality(payoffs, data):
    if not payoffs:
        return True, []

    standalone_payoffs = data.set_index('Farmer_ID')['Standalone_Payoff_INR'].to_dict()
    failing_farmers = []
    all_met = True
    tolerance = 1e-7

    for farmer_id, payoff in payoffs.items():
        if farmer_id not in standalone_payoffs:
            print(f"[!] Farmer {farmer_id} not found in data for IR check.")
            continue

        if payoff < standalone_payoffs[farmer_id] - tolerance:
            all_met = False
            failing_farmers.append(farmer_id)
            # print(f"[-] IR Fail: Farmer {farmer_id}, Payoff={payoff:.2f}, Standalone={standalone_payoffs[farmer_id]:.2f}")


    return all_met, failing_farmers

def calculate_budget_balance(payoffs, buyer_cost):
    if buyer_cost is None or np.isnan(buyer_cost):
         return np.nan
    # return buyer_cost

    total_payments_received = sum(payoffs.values())
    return total_payments_received - buyer_cost
    
    
def calculate_ir_percentage(payoffs: dict, data: pd.DataFrame) -> float:
    n_farmers = len(data)
    if n_farmers == 0: return 100.0
    standalone_payoffs = data.set_index('Farmer_ID')['Standalone_Payoff_INR'].to_dict()
    met_ir_count = 0
    tolerance = 1e-6
    for farmer_id, payoff in payoffs.items():
         if farmer_id in standalone_payoffs:
              if payoff >= standalone_payoffs[farmer_id] - tolerance:
                   met_ir_count += 1
    return (met_ir_count / n_farmers) * 100 if n_farmers > 0 else 100.0
