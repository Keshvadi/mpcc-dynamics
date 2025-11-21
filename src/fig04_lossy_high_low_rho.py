import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def plot_figure_4a():
    """
    Generates Figure 4(a): Type-1 Lossy Equilibrium (High Responsiveness).
    The loss event is absorbed into the existing P-step cycle.
    """
    print("--- Generating Figure 4(a) (Type-1 Lossy, High-rho) ---")

    # --- Parameters ---
    P = 3           # Number of paths
    rho = 0.45      # Responsiveness (rho)
    sigma = 0.9     # Reset softness (sigma)
    N = 1000        # Total number of agents
    C_pi = 12000    # Path capacity
    gamma = 0.7     # Multiplicative decrease factor (gamma)
    
    # Simulation time
    BURN_IN_STEPS = 200
    PLOT_STEPS = 21
    T_MAX = BURN_IN_STEPS + PLOT_STEPS

    # --- 1. Pre-computation ---
    # Calculate standard equilibrium components
    a_hat_p = np.array([((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)])
    alpha_hat_p = np.ones(P) 
    z_rho_p = (1 - (1 - rho)**(P - 1)) / (rho * (1 - rho)**(P - 1))
    q_factor = 1 + rho * sigma * z_rho_p

    # --- 2. Simulate Full Lossy Dynamics ---
    load_history = np.zeros((T_MAX, P))
    path_ranks = np.arange(P)

    for t in range(T_MAX - 1):
        current_load = load_history[t, :]
        next_load = np.zeros(P)
        
        for i in range(P):
            rank = path_ranks[i]
            is_lossy = current_load[i] > C_pi
            
            if rank == P - 1: # Path is pi_min(t) (Target)
                if is_lossy:
                    next_load[i] = (gamma + rho * sigma * z_rho_p) * current_load[i]
                else:
                    next_load[i] = q_factor * current_load[i] + alpha_hat_p[P-1] * a_hat_p[P-1]
            else: # Path is not pi_min(t)
                if is_lossy:
                    next_load[i] = gamma * (1 - rho) * current_load[i]
                else:
                    next_load[i] = (1 - rho) * (current_load[i] + alpha_hat_p[rank] * a_hat_p[rank])
        
        load_history[t+1, :] = next_load
        # Standard P-step rank rotation
        path_ranks = (path_ranks - 1 + P) % P

    # --- 3. Data Extraction for Plotting ---
    plot_data = load_history[BURN_IN_STEPS:, :]
    time_axis = np.arange(PLOT_STEPS)

    # Identify equilibrium features
    L_star_plus = np.max(plot_data)
    L_star_minus = np.min(plot_data)
    
    # Robust peak finding to determine period L
    peaks, _ = find_peaks(plot_data[:, 0], height=L_star_plus * 0.9, distance=P)
    
    if len(peaks) > 1:
        L = peaks[1] - peaks[0]
        peak_time = peaks[0]
    else:
        L = P
        peak_time = 0
    
    f_hat_1_star = plot_data[peak_time, (peak_time + 1) % P]
    f_hat_2_star = plot_data[peak_time, (peak_time + 2) % P]

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    path_colors = ['#0077BB', '#EE7733', '#EE3377']
    
    # Highlight one period
    ax.axvspan(peak_time, peak_time + L, color='lightblue', alpha=0.4)
    
    for i in range(P):
        ax.plot(time_axis, plot_data[:, i], '-o', color=path_colors[i], markersize=5)

    # Reference Lines
    ax.axhline(L_star_plus, ls='--', color='black')
    ax.axhline(C_pi, ls='--', color='gray')
    ax.axhline(f_hat_1_star, ls='--', color='black')
    ax.axhline(gamma * (1 - rho) * C_pi, ls='--', color='black')
    ax.axhline(f_hat_2_star, ls='--', color='black')
    ax.axhline(L_star_minus, ls='--', color='firebrick')
    ax.axhline(gamma * (1 - rho)**2 * C_pi, ls='--', color='black')

    # Annotations
    annot_x = 0.5 
    ax.annotate(r"$\hat{L}^{(0)}$", xy=(annot_x, L_star_plus), xytext=(0, 5), 
                textcoords='offset points', ha='center', va='bottom', fontsize=12)
    ax.annotate(r"Capacity ($C_{\pi}$)", xy=(annot_x, C_pi), xytext=(0, 5), 
                textcoords='offset points', ha='left', va='bottom', fontsize=12, color='gray')
    ax.annotate(r"$\hat{L}^{(P-1)}$", xy=(annot_x, L_star_minus), xytext=(0, -5), 
                textcoords='offset points', ha='center', va='top', fontsize=12, color='firebrick')
    
    # Styling
    ax.set_xlabel('Time Step ($t$)', fontsize=14)
    ax.set_ylabel(r'Expected Path Load ($\hat{L}_{\pi}(t)$)', fontsize=14)
    ax.set_xlim(0, PLOT_STEPS - 1)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
    # Legend
    handles = [plt.Line2D([0], [0], color=c, marker='o') for c in path_colors[:P]]
    labels = [f'Path {i+1} Load' for i in range(P)]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
    plt.tight_layout() 

    # Save
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig04a_lossy_high_rho')
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"  Saved {base_filename}.pdf")


def plot_figure_4b():
    """
    Generates Figure 4(b): Type-2 Lossy Equilibrium (Low Responsiveness).
    The severe multiplicative decrease disrupts the standard rank ordering.
    """
    print("--- Generating Figure 4(b) (Type-2 Lossy, Low-rho) ---")
    
    # --- Parameters ---
    P = 3           # Number of paths
    rho = 0.1       # Responsiveness (rho)
    sigma = 1.0     # Reset softness (sigma)
    N = 1000        # Total number of agents
    C_pi = 12000    # Path capacity
    gamma = 0.7     # Multiplicative decrease factor (gamma)
    
    # Simulation time
    BURN_IN_STEPS = 200
    PLOT_STEPS = 14
    T_MAX = BURN_IN_STEPS + PLOT_STEPS

    # --- 1. Pre-computation ---
    a_hat_p = np.array([((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)])
    alpha_hat_p = np.ones(P)
    z_rho_p = (1 - (1 - rho)**(P - 1)) / (rho * (1 - rho)**(P - 1))
    
    # --- 2. Simulate Dynamics with Dynamic Ranks ---
    load_history = np.zeros((T_MAX, P))
    path_ranks = np.arange(P) 

    for t in range(T_MAX - 1):
        current_load = load_history[t, :]
        next_load = np.zeros(P)
        
        # --- Apply load dynamics based on current ranks ---
        for i in range(P):
            rank = path_ranks[i]
            is_lossy = current_load[i] > C_pi
            is_pi_min = (rank == P - 1)
            
            if is_pi_min:
                q_factor = 1 + rho * sigma * z_rho_p
                if is_lossy:
                    next_load[i] = (gamma + rho * sigma * z_rho_p) * current_load[i]
                else:
                    next_load[i] = q_factor * current_load[i] + alpha_hat_p[P-1] * a_hat_p[P-1]
            else: # Path is not pi_min(t)
                if is_lossy:
                    next_load[i] = gamma * (1 - rho) * current_load[i]
                else:
                    next_load[i] = (1 - rho) * (current_load[i] + alpha_hat_p[rank] * a_hat_p[rank])
        
        load_history[t+1, :] = next_load
        
        # --- Update ranks for the next time step ---
        pi_rank_0_idx = np.where(path_ranks == 0)[0][0]
        loss_occurred = current_load[pi_rank_0_idx] > C_pi
        
        if loss_occurred:
            # Type-2 rule: Overloaded path drops immediately to lowest rank (P-1)
            next_ranks = np.zeros_like(path_ranks)
            next_ranks[pi_rank_0_idx] = P - 1
            for i in range(P):
                if i != pi_rank_0_idx:
                    next_ranks[i] = path_ranks[i] - 1
            path_ranks = next_ranks
        else:
            # Normal P-step oscillation
            path_ranks = (path_ranks - 1 + P) % P

    # --- 3. Data Extraction ---
    plot_data = load_history[BURN_IN_STEPS:, :]
    time_axis = np.arange(PLOT_STEPS)
    
    L_star_plus = np.max(plot_data)
    L_star_minus = np.min(plot_data)

    peaks, _ = find_peaks(plot_data[:, 0], height=L_star_plus * 0.9, distance=P)

    if len(peaks) > 1:
        L = peaks[1] - peaks[0]
        peak_time = peaks[0]
    else:
        L = P 
        peak_time = 0
    
    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    path_colors = ['#0077BB', '#EE7733', '#EE3377']
    
    # Highlight one period
    ax.axvspan(peak_time, peak_time + L, color='lightblue', alpha=0.4)
    
    for i in range(P):
        ax.plot(time_axis, plot_data[:, i], '-o', color=path_colors[i])

    # Reference Lines
    ax.axhline(L_star_plus, ls='--', color='black')
    ax.axhline(C_pi, ls='--', color='gray')
    ax.axhline(L_star_minus, ls='--', color='firebrick')
    ax.axhline(gamma * (1 - rho) * C_pi, ls='--', color='black')
    # Contextual intermediate lines
    ax.axhline(plot_data[peak_time, (peak_time + 1) % P], ls='--', color='black')
    ax.axhline(plot_data[peak_time, (peak_time + 2) % P], ls='--', color='black')

    # Annotations
    annot_x = 0.5 
    ax.annotate(r"$\hat{L}^{(0)}$", xy=(annot_x, L_star_plus), xytext=(0, 5), 
                textcoords='offset points', ha='center', va='bottom', fontsize=12)
    ax.annotate(r"Capacity ($C_{\pi}$)", xy=(annot_x, C_pi), xytext=(0, 5), 
                textcoords='offset points', ha='left', va='bottom', fontsize=12, color='gray')
    ax.annotate(r"$\hat{L}^{(P-1)}$", xy=(annot_x, L_star_minus), xytext=(0, -5), 
                textcoords='offset points', ha='center', va='top', fontsize=12, color='firebrick')

    # Styling
    ax.set_xlabel('Time Step (t)', fontsize=14)
    ax.set_ylabel(r'Expected Path Load ($\hat{L}_{\pi}(t)$)', fontsize=14)
    ax.set_xlim(0, PLOT_STEPS - 1)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
    # Legend
    handles = [plt.Line2D([0], [0], color=c, marker='o') for c in path_colors]
    labels = [f'Path {i+1} Load' for i in range(P)]
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11)
    plt.tight_layout() 

    # Save
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig04b_lossy_low_rho')
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"  Saved {base_filename}.pdf")

def plot_figure_4_all():
    """Generate both lossy equilibrium figures."""
    plot_figure_4a()
    plot_figure_4b()
    print("\nAll Figure 4 plots generated.")
    plt.show()

if __name__ == "__main__":
    plot_figure_4_all()