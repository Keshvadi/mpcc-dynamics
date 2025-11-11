import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

def calculate_continuity_dist(tau, p, P, m, theta):
    """
    Calculates the continuity-time probability distribution using Eq. (37).
    This function is vectorized to handle an array of tau values.
    """
    probs = np.zeros_like(tau, dtype=float)
    
    # Case 2: τ < θ and τ mod P = p
    condition2 = (tau < theta) & (tau % P == p)
    if np.any(condition2):
        exponent2 = np.floor((tau[condition2] - p) / P) * (P - 1)
        probs[condition2] = (1 - (1 - m)**(P - 1)) * (1 - m)**exponent2

    # Case 1: τ = θ
    if not np.isinf(theta):
        condition1 = (tau == theta)
        if np.any(condition1):
             exponent1 = np.ceil((theta - p) / P) * (P - 1)
             probs[condition1] = (1 - m)**exponent1
        
    return probs 

def envelope_function(tau, p, P, m):
    """Calculates the dashed envelope function from the figure's caption."""
    return (1 - (1-m)**(P-1)) * (1-m)**((tau - p)/P * (P-1))

def plot_continuity_distribution_figure():
    """Generates and saves the two-panel continuity distribution figure."""
    
    # --- Parameters ---
    P = 4
    m = 0.12
    tau_range = np.arange(0, 25)

    # --- Plotting Setup ---
    sns.set_style("ticks")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    
    colors = sns.color_palette("deep", P) 
    markers = ['o', 's', '^','*']

    # --- Panel 1: Finite Horizon (θ = 8) ---
    ax1 = axes[0]
    theta1 = 15
    ax1.set_title(fr'$\theta(\pi_i(t), t) = {theta1}$, m = {m}', fontsize=14)

    for p_rank in range(P):
        probs = calculate_continuity_dist(tau_range, p_rank, P, m, theta1)
        ax1.plot(tau_range, probs, marker=markers[p_rank], color=colors[p_rank],
                 linestyle='-', label=f'path {p_rank}')
        
        envelope_tau = np.arange(p_rank, tau_range.max() + 1, P)
        envelope_y = envelope_function(envelope_tau, p_rank, P, m)
        ax1.plot(envelope_tau, envelope_y, linestyle='-.', color=colors[p_rank], alpha=0.6)

    # --- Panel 2: Infinite Horizon (θ -> ∞) ---
    ax2 = axes[1]
    theta2 = float('inf')
    ax2.set_title(r'$\theta(\pi_i(t), t) \rightarrow \infty$', fontsize=14)

    for p_rank in range(P):
        probs = calculate_continuity_dist(tau_range, p_rank, P, m, theta2)
        ax2.plot(tau_range, probs, marker=markers[p_rank], color=colors[p_rank],
                 linestyle='-')
        
        envelope_tau = np.arange(p_rank, tau_range.max() + 1, P)
        envelope_y = envelope_function(envelope_tau, p_rank, P, m)
        ax2.plot(envelope_tau, envelope_y, linestyle='-.', color=colors[p_rank], alpha=0.6)

    # --- Final Styling and Metadata ---
    fig.supylabel(r'Probability Mass $\mathbb{P}[\tau_i(t) = \tau \mid \mathrm{rank}(\pi_i(t),t) = p]$', fontsize=14)
    
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)

    ax2.set_xlabel(r'Agent Continuity Time ($\tau$)', fontsize=14)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax1.grid(True, which='both', linestyle=':')
    ax2.grid(True, which='both', linestyle=':')
    
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), fontsize=12)
    plt.tight_layout()
    
    # --- Create output directory and save figures ---
    output_dir = '../figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig01_continuity_dist')
    
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"Plot saved as '{base_filename}.png' and '{base_filename}.pdf'")


if __name__ == "__main__":
    plot_continuity_distribution_figure()
