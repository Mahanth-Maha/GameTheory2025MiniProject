import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

import best_config
from utils import run_vcg_auction
from metrics import gini_coefficient, vcg_individual_rationality, vcg_budget_balance

def generate_supply_curve(data, price_range):
    print(f"\n--- Running Supply Curve Generation (Prices: {min(price_range)} to {max(price_range)}) ---\n")
    supply_data = []
    for price in price_range:
        winners, payments, surplus, total_payments = run_vcg_auction(data, price_per_credit=price)
        total_credits = 0
        if winners:
            total_credits = data[data['Farmer_ID'].isin(winners)]['Potential_Carbon_Credits_tCO2e'].sum()

        supply_data.append({
            'Price_per_Credit': price,
            'Total_Credits_Supplied': total_credits,
            'Number_of_Suppliers': len(winners),
            'Total_Surplus': surplus,
            'Total_Payments': total_payments
        })
    results_df = pd.DataFrame(supply_data)
    print(f"[>] Supply Curve Data Generated:")
    print(results_df.round(2))
    return results_df

def analyze_vcg_outcomes(data, price_range):
    """ RQ 2d: Analyze VCG outcomes across different prices. """
    print(f"\n--- Running VCG Outcome Analysis (Prices: {min(price_range)} to {max(price_range)}) ---\n")
    vcg_results = []
    for price in price_range:
        winners, payments, surplus, total_payments = run_vcg_auction(data, price_per_credit=price)
        num_winners = len(winners)
        payments_gini = gini_coefficient(list(payments.values()))
        is_ir_met = vcg_individual_rationality(payments, data) # Pass full data for cost lookup

        total_credits_allocated = 0
        if winners:
             total_credits_allocated = data[data['Farmer_ID'].isin(winners)]['Potential_Carbon_Credits_tCO2e'].sum()
        total_buyer_value = price * total_credits_allocated
        budget_bal = vcg_budget_balance(total_payments, total_buyer_value)

        vcg_results.append({
            'Price_per_Credit': price,
            'Num_Winners': num_winners,
            'Total_Surplus': surplus,
            'Total_Payments': total_payments,
            'Payments_Gini': payments_gini,
            'IR_Met': is_ir_met,
            'Budget_Balance': budget_bal
        })

    results_df = pd.DataFrame(vcg_results)
    print(f"[>] VCG Outcome Analysis Results:")
    print(results_df.round(2))
    return results_df

def plot_supply_curve(df, output_dir):
    if df is None or df.empty: 
        return
    plt.figure(figsize=(8, 5))
    plt.plot(df['Total_Credits_Supplied'], df['Price_per_Credit'], marker='o', linestyle='-')
    plt.xlabel("Total Carbon Credits Supplied (tCO2e)")
    plt.ylabel("Price per Credit (INR)")
    plt.title("VCM Supply Curve")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = os.path.join(output_dir, 'vcm_supply_curve.png')
    plt.savefig(filename)
    print(f"[>] Saved supply curve plot: {filename}")
    plt.close()

def plot_vcg_outcomes(df, output_dir):
    if df is None or df.empty: return
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)
    ax1 = axes[0]
    ax1.plot(df['Price_per_Credit'], df['Num_Winners'], marker='o', linestyle='-', label='Num Winners', color='blue')
    ax1.set_ylabel('Number of Winners', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Price_per_Credit'], df['Total_Surplus'], marker='s', linestyle='--', label='Total Surplus', color='green')
    ax1_twin.set_ylabel('Total Surplus (INR)', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.set_title('VCG Winners & Surplus vs. Price')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1_twin.legend(lines + lines2, labels + labels2, loc='upper left')


    
    axes[1].plot(df['Price_per_Credit'], df['Payments_Gini'], marker='o', linestyle='-')
    axes[1].set_ylabel('Payments Gini Coefficient')
    axes[1].set_title('VCG Payments Gini vs. Price')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(df['Price_per_Credit'], df['Budget_Balance'], marker='o', linestyle='-')
    axes[2].set_ylabel('Budget Balance (INR)')
    axes[2].set_xlabel('Price per Credit (INR)')
    axes[2].set_title('VCG Budget Balance vs. Price (Payment - Value)')
    axes[2].axhline(0, color='red', linestyle='--', linewidth=0.8, label='Balanced')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    filename = os.path.join(output_dir, 'vcg_outcomes_vs_price.png')
    plt.savefig(filename)
    print(f"[>] Saved VCG outcomes plot: {filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pricing/Trading Analysis Experiments")
    parser.add_argument('-f', "--file", type=str, default=best_config.DEFAULT_DATA_FILE, help="Path to the dataset file.")
    parser.add_argument('-n', "--n_farmers", type=int, default=best_config.N_PRICING_ANALYSIS, help="Number of farmers to sample.")
    parser.add_argument('--seed', type=int, default=best_config.RANDOM_SEED, help="Random seed for sampling.")
    parser.add_argument('--price_min', type=float, default=500.0, help="Minimum price for analysis range.")
    parser.add_argument('--price_max', type=float, default=4000.0, help="Maximum price for analysis range.")
    parser.add_argument('--price_steps', type=int, default=15, help="Number of price steps in the range.")
    parser.add_argument('--plot_dir', type=str, default=os.path.join(best_config.PLOT_OUTPUT_DIR, 'pricing/'), help="Directory for saving plots.")
    args = parser.parse_args()
    
    args.plot_dir = os.path.join(args.plot_dir, f"n_farmers_{args.n_farmers}/")
    
    os.makedirs(args.plot_dir, exist_ok=True)

    try:
        full_data = pd.read_csv(args.file)
        if args.n_farmers < len(full_data):
             data = full_data.sample(n=args.n_farmers, random_state=args.seed).reset_index(drop=True)
             print(f"[>] Sampled {args.n_farmers} farmers from {args.file}")
        else:
             data = full_data
             print(f"[>] Using all {len(data)} farmers from {args.file}")
    except FileNotFoundError:
        print(f"[-] Data file not found: {args.file}")
        exit()

    price_range = np.linspace(args.price_min, args.price_max, args.price_steps)
    
    supply_curve_df = generate_supply_curve(data, price_range)
    plot_supply_curve(supply_curve_df, args.plot_dir)
    supply_curve_df.to_csv(os.path.join(args.plot_dir, 'supply_curve_data.csv'), index=False)

    vcg_outcomes_df = analyze_vcg_outcomes(data, price_range)
    plot_vcg_outcomes(vcg_outcomes_df, args.plot_dir)
    vcg_outcomes_df.to_csv(os.path.join(args.plot_dir, 'vcg_outcomes_data.csv'), index=False)