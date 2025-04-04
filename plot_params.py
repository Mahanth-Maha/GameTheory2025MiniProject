import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def save_plot(filename, output_dir):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    print(f"[>] Saved plot: {filepath}")
    plt.close()

def plot_results(log_file, output_dir):
    if not os.path.exists(log_file):
        print(f"[-] Log file not found: {log_file}")
        return

    try:
        df = pd.read_csv(log_file)
        print(f"[>] Loaded {len(df)} results from {log_file}")
    except Exception as e:
        print(f"[-] Error loading log file: {e}")
        return

    if 'status' in df.columns:
        df = df[df['status'] != 'failed'].copy()
        if df.empty:
            print(f"[!] No successful runs found in the log file. Cannot generate plots.")
            return
        print(f"[>] Filtered to {len(df)} successful runs.")

    os.makedirs(output_dir, exist_ok=True)

    # VCG Surplus vs. VCG Price
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='vcg_price_per_credit', y='vcg_total_surplus', marker='o', errorbar='sd')
    plt.title('VCG Total Surplus vs. Price per Credit (Mean +/- SD)')
    plt.xlabel('VCG Price per Credit (INR)')
    plt.ylabel('Total Social Surplus (INR)')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('vcg_surplus_vs_price.png', output_dir)

    #  Shapley Gini vs. Synergy Parameters (alpha, beta)
    if 'alpha' in df.columns and 'beta' in df.columns:
         g = sns.FacetGrid(df, col="beta", height=4, aspect=1.2)
         g.map(sns.lineplot, "alpha", "shapley_values_gini", marker='o', errorbar="sd")
         g.add_legend()
         g.fig.suptitle('Shapley Gini vs. Alpha (faceted by Beta)', y=1.03)
         g.set_axis_labels("Alpha (Synergy Multiplier)", "Shapley Gini Coefficient")
         save_plot('shapley_gini_vs_alpha_beta.png', output_dir)

    # VCG Payments Gini vs. VCG Price
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='vcg_price_per_credit', y='vcg_payments_gini', marker='o', errorbar='sd')
    plt.title('VCG Payments Gini vs. Price per Credit (Mean +/- SD)')
    plt.xlabel('VCG Price per Credit (INR)')
    plt.ylabel('VCG Payments Gini Coefficient')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('vcg_gini_vs_price.png', output_dir)

    # Trade-off: VCG Surplus vs. Shapley Gini
    plt.figure(figsize=(10, 7))
    hue_col = 'is_shapley_in_core' if 'is_shapley_in_core' in df.columns and df['core_check_performed'].any() else None
    style_col = 'utility_func_name' if 'utility_func_name' in df.columns else None

    sns.scatterplot(data=df, x='shapley_values_gini', y='vcg_total_surplus', hue=hue_col, style=style_col, alpha=0.8, s=80)
    plt.title('Efficiency vs. Equity Trade-off (VCG Surplus vs. Shapley Gini)')
    plt.xlabel('Shapley Allocation Gini (Lower is more equal)')
    plt.ylabel('VCG Total Surplus (Higher is more efficient)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title=f'Hue: Core Stable | Style: Utility' if hue_col and style_col else 'Parameters')
    save_plot('tradeoff_surplus_vs_shapley_gini.png', output_dir)

    # Core Stability Analysis
    if 'is_shapley_in_core' in df.columns and df['core_check_performed'].any():
        core_checked_df = df[df['core_check_performed'] == True]
        if not core_checked_df.empty:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=core_checked_df, x='is_shapley_in_core', hue='utility_func_name', dodge=True)
            plt.title('Core Stability of Shapley Allocation (for runs where Core was checked)')
            plt.xlabel('Is Shapley Allocation in Core?')
            plt.ylabel('Number of Parameter Combinations')
            plt.xticks([False, True], ['Not in Core', 'In Core'])
            save_plot('core_stability_counts.png', output_dir)

            core_failed_df = core_checked_df[core_checked_df['is_shapley_in_core'] == False]
            if not core_failed_df.empty and 'blocking_coalitions_count' in core_failed_df.columns:
                 plt.figure(figsize=(10, 6))
                 sns.boxplot(data=core_failed_df, x='alpha', y='blocking_coalitions_count', hue='beta')
                 plt.title('Number of Blocking Coalitions vs. Synergy (when Shapley not in Core)')
                 plt.xlabel('Alpha')
                 plt.ylabel('Count of Blocking Coalitions')
                 plt.legend(title='Beta', loc='upper right')
                 save_plot('blocking_coalitions_vs_synergy.png', output_dir)


    # VCG Winners vs Price
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='vcg_price_per_credit', y='vcg_num_winners', marker='o', errorbar='sd')
    plt.title('Number of VCG Winners vs. Price per Credit (Mean +/- SD)')
    plt.xlabel('VCG Price per Credit (INR)')
    plt.ylabel('Number of Winning Farmers')
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('vcg_winners_vs_price.png', output_dir)

    print(f"[>] Plots saved in directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from parameter search log file.")
    parser.add_argument('-l', "--log_file", type=str, default='./data/synthetic/parameters_check.csv', help="Path to the parameter CSV log file.")
    parser.add_argument('-o', "--output_dir", type=str, default='./logs/plots/', help="Directory to save the generated plots.")
    args = parser.parse_args()

    plot_results(args.log_file, args.output_dir)
    
# python plot_params.py
