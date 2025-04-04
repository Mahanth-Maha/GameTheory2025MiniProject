import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_mechanism_comparison(results_df, varying_param, output_dir = './logs/plots/mechanism/'):
    os.makedirs(output_dir, exist_ok=True)
    mechanisms = results_df['mechanism_name'].unique()

    # Average Farmer Profit
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x=varying_param, y='avg_farmer_profit', hue='mechanism_name', marker='o')
    plt.title(f'Average Farmer Profit vs. {varying_param}')
    plt.ylabel('Average Profit (INR)')
    plt.xlabel(varying_param.replace('_', ' ').title())
    plt.legend(title='Mechanism')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'avg_profit_vs_{varying_param}.png'))
    plt.close()

    # Gini Coefficient (Fairness)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x=varying_param, y='gini_coefficient', hue='mechanism_name', marker='o')
    plt.title(f'Fairness (Gini Coefficient) vs. {varying_param}')
    plt.ylabel('Gini Coefficient (0=Perfect Equality)')
    plt.xlabel(varying_param.replace('_', ' ').title())
    plt.legend(title='Mechanism')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'gini_vs_{varying_param}.png'))
    plt.close()

    # Percentage of Farmers Meeting Individual Rationality
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x=varying_param, y='ir_met_percentage', hue='mechanism_name', marker='o')
    plt.title(f'Individual Rationality Met (%) vs. {varying_param}')
    plt.ylabel('% Farmers Meeting IR')
    plt.xlabel(varying_param.replace('_', ' ').title())
    plt.ylim(0, 101)
    plt.legend(title='Mechanism')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ir_met_vs_{varying_param}.png'))
    plt.close()

    # Buyer Cost (for auction mechanisms)
    auction_results = results_df[results_df['buyer_cost'].notna()]
    if not auction_results.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=auction_results, x=varying_param, y='buyer_cost', hue='mechanism_name', marker='o')
        plt.title(f'Total Buyer Cost vs. {varying_param}')
        plt.ylabel('Total Buyer Cost (INR)')
        plt.xlabel(varying_param.replace('_', ' ').title())
        plt.legend(title='Mechanism')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'buyer_cost_vs_{varying_param}.png'))
        plt.close()

    # Core Stability (for Shapley)
    shapley_results = results_df[results_df['mechanism_name'] == 'Shapley Allocation']
    if not shapley_results.empty and 'is_in_core_numeric' in shapley_results.columns:
         shapley_results_core_checked = shapley_results[shapley_results['core_check_performed'] == True]
         if not shapley_results_core_checked.empty:
             plt.figure(figsize=(10, 6))
             sns.lineplot(data=shapley_results_core_checked, x=varying_param, y='is_in_core_numeric', marker='o')
             plt.title(f'Core Stability (Shapley Allocation) vs. {varying_param}')
             plt.ylabel('Is in Core? (1=Yes, 0=No)')
             plt.xlabel(varying_param.replace('_', ' ').title())
             plt.yticks([0, 1])
             plt.grid(True, linestyle='--', alpha=0.6)
             plt.tight_layout()
             plt.savefig(os.path.join(output_dir, f'core_stability_vs_{varying_param}.png'))
             plt.close()


    print(f"[+] Plots saved to {output_dir}")