import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

def calculate_continuity_dist(tau, p, P, rho, theta):
    """
    Calculates the continuity-time probability distribution:
    P[tau_i(t) = tau | rank(pi, t) = p]
    """
    probs = np.zeros_like(tau, dtype=float)
    
    # Case 2: Standard shifting behavior (tau < theta)
    condition2 = (tau < theta) & (tau % P == p)
    if np.any(condition2):
        exponent2 = np.floor((tau[condition2] - p) / P) * (P - 1)
        probs[condition2] = (1 - (1 - rho)**(P - 1)) * (1 - rho)**exponent2

    # Case 1: Spike at loss horizon (tau = theta)
    if not np.isinf(theta):
        condition1 = (tau == theta)
        if np.any(condition1):
             exponent1 = np.ceil((theta - p) / P) * (P - 1)
             probs[condition1] = (1 - rho)**exponent1
        
    return probs 

def envelope_function(tau, p, P, rho):
    """Calculates the theoretical decay envelope for visualization."""
    return (1 - (1-rho)**(P-1)) * (1-rho)**((tau - p)/P * (P-1))

def plot_figure_1():
    """
    Generates Figure 1: Probability distribution of agent continuity times.
    """
    
    # --- Model Parameters ---
    P = 4               # Total Paths
    rho = 0.12          # Responsiveness (Migration Probability)
    tau_range = np.arange(0, 25)
    
    # --- Plot Setup ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True)
    colors = sns.color_palette("deep", P) 
    markers = ['o', 's', '^', 'D'] 

    # --- Panel 1: Finite Horizon (Recent Loss) ---
    ax1 = axes[0]
    theta_finite = 15
    ax1.set_title(fr'$\theta(\pi, t) = {theta_finite}$, $\rho = {rho}$', fontsize=16)

    for p_rank in range(P):
        probs = calculate_continuity_dist(tau_range, p_rank, P, rho, theta_finite)
        
        ax1.plot(tau_range, probs, marker=markers[p_rank], color=colors[p_rank],
                 linestyle='-', label=f'Rank $p={p_rank}$')
        
        # Theoretical envelope
        envelope_tau = np.arange(p_rank, tau_range.max() + 1, P)
        envelope_y = envelope_function(envelope_tau, p_rank, P, rho)
        ax1.plot(envelope_tau, envelope_y, linestyle='-.', color=colors[p_rank], alpha=0.5, linewidth=1)

    # --- Panel 2: Infinite Horizon (Steady State) ---
    ax2 = axes[1]
    theta_inf = float('inf')
    ax2.set_title(r'$\theta(\pi, t) \rightarrow \infty$ (Steady State)', fontsize=16)

    for p_rank in range(P):
        probs = calculate_continuity_dist(tau_range, p_rank, P, rho, theta_inf)
        ax2.plot(tau_range, probs, marker=markers[p_rank], color=colors[p_rank],
                 linestyle='-')
        
        # Theoretical envelope
        envelope_tau = np.arange(p_rank, tau_range.max() + 1, P)
        envelope_y = envelope_function(envelope_tau, p_rank, P, rho)
        ax2.plot(envelope_tau, envelope_y, linestyle='-.', color=colors[p_rank], alpha=0.5, linewidth=1)

    # --- Formatting & Labels ---
    fig.supylabel(r'Probability Mass $\mathbb{P}[\tau_i(t) = \tau \mid \mathrm{rank}(\pi,t) = p]$', fontsize=16)
    
    ax1.set_ylim(0, 0.5)
    ax2.set_ylim(0, 0.5)

    ax2.set_xlabel(r'Agent Continuity Time ($\tau$)', fontsize=16)
    ax2.set_xticks(np.arange(0, 25, 2))
    
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig01_continuity_dist')
    
    # Added bbox_inches='tight'
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'{base_filename}.pdf', bbox_inches='tight', pad_inches=0.05)
    
    print(f"Figure 1 saved to: {base_filename}.pdf")
    plt.show()

if __name__ == "__main__":
    plot_figure_1()