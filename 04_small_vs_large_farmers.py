import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import characteristic_function_v, shapley_value_exact, shapley_value_monte_carlo

def generate_heterogeneous_data(n_small=10, n_large=5, seed=24004):
    np.random.seed(seed)
    farmers = []
    total_farmers = n_small + n_large

    for i in range(n_small):
        farmer_id = f'F_S_{i:03d}'
        r_i = np.round(np.random.normal(15000, 3000), -2).clip(5000)
        q_i = np.round(np.random.gamma(2.0, 1.0), 2).clip(0.1)
        c_i = np.round(np.random.gamma(3.5, 900) + 600, -2).clip(100)
        farmers.append({'Farmer_ID': farmer_id, 'Type': 'Small', 'Standalone_Payoff_INR': r_i,'Potential_Carbon_Credits_tCO2e': q_i,'True_Cost_per_Credit_INR': c_i})

    for i in range(n_large):
        farmer_id = f'F_L_{i:03d}'
        r_i = np.round(np.random.normal(40000, 8000), -2).clip(15000)
        q_i = np.round(np.random.gamma(3.0, 2.5), 2).clip(0.5)
        c_i = np.round(np.random.gamma(2.5, 700) + 400, -2).clip(100)
        farmers.append({'Farmer_ID': farmer_id, 'Type': 'Large','Standalone_Payoff_INR': r_i,'Potential_Carbon_Credits_tCO2e': q_i,'True_Cost_per_Credit_INR': c_i})
    
    return pd.DataFrame(farmers)

N_SMALL = 10
N_LARGE = 5
N_TOTAL = N_SMALL + N_LARGE
SEED = 24004
ALPHA = 1.25
BETA = 0.0
PLOT_DIR = './logs/plots/heterogeneity/'
os.makedirs(PLOT_DIR, exist_ok=True)

print(f"[>] Generating data for {N_SMALL} small and {N_LARGE} large farmers...")
data = generate_heterogeneous_data(n_small=N_SMALL, n_large=N_LARGE, seed=SEED)
print(data)

farmer_ids = data['Farmer_ID'].tolist()
gt_params = {'alpha': ALPHA, 'beta': BETA}

print(f"\n[>] Calculating Shapley values for N={N_TOTAL} (Alpha={ALPHA}, Beta={BETA})...", end = '')
if N_TOTAL <= 10:
    shapley_values = shapley_value_exact(farmer_ids, data, characteristic_function_v, gt_params)
    print("   (Using Exact Method)")
else:
    shapley_values = shapley_value_monte_carlo(farmer_ids, data, characteristic_function_v, gt_params, n_samples=1000*N_TOTAL)
    print(f"   (Using Monte Carlo with {1000*N_TOTAL} samples)")

results = []
for farmer_id in farmer_ids:
    row = data[data['Farmer_ID'] == farmer_id].iloc[0]
    r_i = row['Standalone_Payoff_INR']
    phi_i = shapley_values.get(farmer_id, 0)
    gain_abs = phi_i - r_i
    gain_pct = (gain_abs / r_i) * 100 if r_i > 0 else 0
    results.append({'Farmer_ID': farmer_id,'Type': row['Type'],
        'Standalone_Payoff': r_i,
        'Shapley_Value': phi_i,
        'Absolute_Gain': gain_abs,
        'Percentage_Gain': gain_pct
    })

results_df = pd.DataFrame(results)
print("\n[+] Results:")
print(results_df.round(2))

avg_results = results_df.groupby('Type').agg(
    Avg_Standalone_Payoff=('Standalone_Payoff', 'mean'),
    Avg_Shapley_Value=('Shapley_Value', 'mean'),
    Avg_Absolute_Gain=('Absolute_Gain', 'mean'),
    Avg_Percentage_Gain=('Percentage_Gain', 'mean')
).reset_index()

print("\n[>] Average Results by Farmer Type:")
print(avg_results.round(2))

print(f"\n[>] Generating plots in {PLOT_DIR}...")
plt.figure(figsize=(8, 6))
colors = {'Small': 'blue', 'Large': 'red'}
for f_type in results_df['Type'].unique():
    subset = results_df[results_df['Type'] == f_type]
    plt.scatter(subset['Standalone_Payoff'], subset['Shapley_Value'],
                label=f_type, color=colors[f_type], alpha=0.7)
plt.plot([data['Standalone_Payoff_INR'].min(), data['Standalone_Payoff_INR'].max()],
         [data['Standalone_Payoff_INR'].min(), data['Standalone_Payoff_INR'].max()],
         'k--', label='Break-even (Gain=0)')
plt.xlabel('Standalone Payoff (INR)')
plt.ylabel('Shapley Value (INR)')
plt.title(f'Shapley Value vs. Standalone Payoff (N={N_TOTAL}, Alpha={ALPHA}, Beta={BETA})')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'shapley_vs_standalone_hetero.png'))
plt.close()
print("[+] Saved shapley_vs_standalone_hetero.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
avg_results.plot(kind='bar', x='Type', y='Avg_Absolute_Gain', ax=axes[0], legend=False, color=['blue', 'red'])
axes[0].set_title('Average Absolute Gain (Shapley - Standalone)')
axes[0].set_ylabel('Gain (INR)')
axes[0].tick_params(axis='x', rotation=0)
avg_results.plot(kind='bar', x='Type', y='Avg_Percentage_Gain', ax=axes[1], legend=False, color=['blue', 'red'])
axes[1].set_title('Average Percentage Gain')
axes[1].set_ylabel('Gain (%)')
axes[1].tick_params(axis='x', rotation=0)
plt.suptitle(f'Comparison of Gains by Farmer Type (Alpha={ALPHA}, Beta={BETA})')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(PLOT_DIR, 'average_gains_hetero.png'))
plt.close()
print("[+] Saved average_gains_hetero.png")