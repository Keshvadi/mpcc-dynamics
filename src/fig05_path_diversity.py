import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def simulate_path_diversity(P, N, rho, sigma, gamma, C_total, alpha, T_max, T_burn):
    """
    Simulates the network dynamics for a specific path count P to measure stability metrics.
    """
    C_per_path = C_total / P
    
    # Initialization (Start slightly perturbed to avoid perfect symmetry lock)
    a = np.ones(P) * (N / P)
    L = np.ones(P) * (C_per_path * 0.5) 
    L[0] *= 1.01 # Small perturbation
    
    L_history = []

    for t in range(T_max):
        # 1. Identify path statuses (Greedy selection)
        idx_min = np.argmin(L)
        is_min = np.zeros(P, dtype=bool)
        is_min[idx_min] = True
        is_others = ~is_min
        
        # 2. Calculate Flows (based on state at t)
        flow_agents_out = a[is_others] * rho
        flow_load_out = L[is_others] * rho
        
        # 3. Update State (to t+1)
        a_next = a.copy()
        L_next = L.copy()
        
        # Agent Migration
        a_next[is_others] -= flow_agents_out
        a_next[idx_min] += np.sum(flow_agents_out)
        
        # Load Migration (Reset Softness applied to incoming load)
        L_next[is_others] -= flow_load_out
        L_next[idx_min] += np.sum(flow_load_out) * sigma
        
        # 4. Congestion Control Dynamics
        # Additive Increase
        L_next += alpha * a_next
        
        # Multiplicative Decrease
        overloaded = L_next > C_per_path
        L_next[overloaded] *= gamma
        
        # Update main variables
        a = a_next
        L = L_next
        
        if t >= T_burn:
            L_history.append(L.copy())

    # Analysis
    L_history = np.array(L_history)
    
    # Global metrics across the burn-in window
    flat_L = L_history.flatten()
    min_L = np.min(flat_L)
    max_L = np.max(flat_L)
    
    # Convergence: Ratio of min to max load (1.0 = perfect stability)
    gamma_conv = min_L / max_L if max_L > 0 else 0
    
    # Efficiency: Lowest utilization relative to capacity
    epsilon = min_L / C_per_path
    
    # Loss Severity: Max overshoot relative to capacity
    lambda_metric = max(0.0, (max_L - C_per_path) / C_per_path)
    
    # Oscillation Amplitude
    amplitude = (max_L - min_L) / C_per_path
    
    return epsilon, gamma_conv, lambda_metric, amplitude

def plot_figure_5():
    """
    Generates Figure 5: Stability Metrics vs. Path Diversity (P).
    """
    print("--- Generating Figure 5 (Path Diversity) ---")
    
    # Parameters
    Ps = np.array([2, 3, 4, 5, 8, 10, 15, 20, 30, 50, 75, 100])
    N = 5000.0          # Total Agents
    rho = 0.15          # Responsiveness (rho)
    sigma = 0.5         # Reset softness (sigma)
    gamma = 0.5         # Multiplicative decrease (gamma)
    C_total = 50000.0   # Total network capacity
    alpha = 1.0         # Additive increase
    T_max = 3000
    T_burn = 1500
    
    print(f"Simulating {len(Ps)} scenarios...")
    results = []
    for p in Ps:
        # Cast p to int for simulation logic
        res = simulate_path_diversity(int(p), N, rho, sigma, gamma, C_total, alpha, T_max, T_burn)
        results.append(res)
    print("Simulations complete.")

    # Unpack results
    results = np.array(results)
    epsilon = results[:, 0]
    gamma_conv = results[:, 1]
    lambda_metric = results[:, 2]
    amplitude = results[:, 3]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Colors consistent with previous figures
    color_conv = '#0077BB' # Blue
    color_eff = '#2CA02C'  # Green
    color_loss = '#D62728' # Red
    color_amp = '#9467BD'  # Purple

    # Panel 1: Stability & Efficiency (Higher is Better)
    ax1.plot(Ps, gamma_conv, marker='o', color=color_conv, label=r'Convergence ($\gamma_{\mathrm{conv}}$)', lw=2)
    ax1.plot(Ps, epsilon, marker='s', color=color_eff, label=r'Efficiency ($\epsilon$)', lw=2)
    
    ax1.set_ylabel('Axiomatic Rating', fontsize=14)
    ax1.grid(True, which="both", linestyle=':', alpha=0.7)
    ax1.legend(loc='best', fontsize=12)

    # Panel 2: Instability & Loss (Lower is Better)
    ax2.plot(Ps, lambda_metric, marker='^', color=color_loss, label=r'Loss Severity ($\lambda$)', lw=2)
    ax2.plot(Ps, amplitude, marker='x', color=color_amp, label='Oscillation Amplitude', lw=2)
    
    ax2.set_xlabel('Number of Paths ($P$)', fontsize=14)
    ax2.set_ylabel('Metric Value', fontsize=14)
    
    # Handle Log Scale and Ticks
    ax2.set_xscale('log')
    ax2.set_xticks(Ps)
    ax2.set_xticklabels(Ps.astype(int)) 
    
    ax2.grid(True, which="both", linestyle=':', alpha=0.7)
    ax2.legend(loc='best', fontsize=12)

    # --- File Saving ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig05_path_diversity')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Figure 5 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_5()