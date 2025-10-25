import matplotlib.pyplot as plt

def apply_springer_sans_serif_theme():
    plt.style.use('default')
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,

        # Titles
        "axes.titlesize": 15,
        # "axes.titleweight": "regular",
        "axes.titleweight": "bold",

        # Axes labels
        "axes.labelsize": 12,
        "axes.labelcolor": "#333333",
        "axes.labelweight": "bold",
        
        # Ticks
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        # Spines and axes
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 1.0,

        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Legends
        "legend.fontsize": 12,
        "legend.title_fontsize": 14,
        "legend.edgecolor": "#333333",

        # Lines/Markers
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1,
        # "lines.markeredgecolor": "black",
        "lines.markerfacecolor": "auto",

        # Color cycle (Altair category palette)
        "axes.prop_cycle": plt.cycler(color=["#1f77b4", "#6e0426", "#008000", "#c66300" ,
                                             "#420eff", "#60b924", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#d42109"]),
        # Savefig
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white"
    })

# apply_springer_sans_serif_theme()

def plot_X_vs_Y(x_vals, y_vals, title, x_label, y_label, save_path="plot.png"):
    apply_springer_sans_serif_theme()
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='#1f77b4', label='Data')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.grid(True, color="#EEEEEE")
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.9), frameon=False)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    plt.close()

def plot_side_by_side(x_vals, y_vals1, y_vals2, title1, title2, x_label1, x_label2, y_label1, y_label2, save_path="side_by_side_plot.png"):
    apply_springer_sans_serif_theme()
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals1, marker='o', linestyle='-', color='#1f77b4', label='Data 1')
    plt.xlabel(x_label1)
    plt.ylabel(y_label1)
    plt.title(title1)
    plt.grid(True, color="#EEEEEE")
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals2, marker='o', linestyle='-', color='#6e0426', label='Data 2')
    plt.xlabel(x_label2)
    plt.ylabel(y_label2)
    plt.title(title2)
    plt.grid(True, color="#EEEEEE")

    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    plt.legend()
    plt.tight_layout(pad=2)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    plt.close()
    
def plot_bar_graph(x_vals, y_vals, title, x_label, y_label, save_path="bar_plot.png"):
    apply_springer_sans_serif_theme()
    plt.figure(figsize=(10, 6))
    plt.bar(x_vals, y_vals, color='#1f77b4', edgecolor="#001e33")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.xticks(rotation=45)
    # plt.grid(axis='y', color="#EEEEEE")
    plt.gca().yaxis.grid(False)
    plt.gca().xaxis.grid(False)

    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.9), frameon=False)

    plt.tight_layout(pad=2)
    plt.savefig(save_path, bbox_inches='tight', dpi=1200)
    plt.close()
    
if __name__ == "__main__":
    # Example usage
    x_vals = [0, 1, 2, 3, 4, 5]
    y_vals = [10, 20, 25, 30, 40, 50]
    
    plot_X_vs_Y(x_vals, y_vals, "Example Plot", "X Axis", "Y Axis", save_path="example_plot.png")
    plot_side_by_side(x_vals, y_vals, y_vals[::-1], "Plot 1", "Plot 2", "X Axis 1", "X Axis 2", "Y Axis 1", "Y Axis 2", save_path="side_by_side_example.png")
    plot_bar_graph(x_vals, y_vals, "Bar Graph Example", "X Axis", "Y Axis", save_path="bar_graph_example.png")
    