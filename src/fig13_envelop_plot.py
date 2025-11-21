import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def run_simulation(P, steps=300, rho=0.1, sigma=0.5, agents_per_path=100, k_subset=5):
    """
    Runs simulation with TRUE 'k-subset' probing to match the paper's claims.
    """
    N = P * agents_per_path
    C_total = N * 10
    C_path = C_total / P
    
    agents_path = np.random.randint(0, P, size=N)
    agents_window = np.ones(N) * 5.0
    
    transient_cutoff = 100
    history_spread = []
    history_fairness = []
    
    for t in range(steps):
        # 1. Calc Loads
        path_loads = np.bincount(agents_path, weights=agents_window, minlength=P)
        utilization = path_loads / C_path
        
        if t > transient_cutoff:
            spread = np.max(utilization) - np.min(utilization)
            history_spread.append(spread)
            history_fairness.append(np.var(agents_window))
        
        # 2. Identify Targets (True Probing Logic)
        # Add noise to break ties in comparison
        noisy_loads = path_loads + np.random.normal(0, C_path * 0.01, P)
        
        # Generate random probe indices for all agents: Shape (N, k_subset)
        # Each row represents the subset of paths a single agent "sees"
        probes = np.random.randint(0, P, size=(N, k_subset))
        
        # Look up the noisy loads for these probes
        probe_loads = noisy_loads[probes]  # Shape (N, k_subset)
        
        # Each agent identifies the best path *within their specific probe set*
        local_best_indices = np.argmin(probe_loads, axis=1)
        
        # Map the local index (0..k) back to the global Path ID (0..P)
        specific_best_path = probes[np.arange(N), local_best_indices]
        
        # 3. Migration Logic
        # Compare the load of the agent's CURRENT path to the path they FOUND
        current_path_loads = noisy_loads[agents_path]
        found_path_loads = noisy_loads[specific_best_path]
        
        # Agent considers moving ONLY if the found path is strictly better
        is_better_path_found = found_path_loads < current_path_loads
        
        # Apply Responsiveness (rho)
        # Migrate if: (Found Better Path) AND (Dice Roll < rho)
        should_migrate = is_better_path_found & (np.random.random(N) < rho)
        
        migrating_indices = np.where(should_migrate)[0]

        # Move agents
        if len(migrating_indices) > 0:
            # Apply Penalties to migrants
            agents_window[migrating_indices] *= sigma
            # Update locations
            agents_path[migrating_indices] = specific_best_path[migrating_indices]
        
        # 4. Congestion Control
        current_loads = np.bincount(agents_path, weights=agents_window, minlength=P)
        overloaded_paths = np.where(current_loads > C_path)[0]
        
        is_overloaded = np.zeros(P, dtype=bool)
        is_overloaded[overloaded_paths] = True
        agents_in_loss = is_overloaded[agents_path]
        
        agents_window[agents_in_loss] *= 0.5
        agents_window[~agents_in_loss] += 1.0
        
    return np.mean(history_spread), np.mean(history_fairness)

def plot_figure_13():
    """
    Generates Figure 13: The Benefit of Scale (Envelope Plot).
    Shows how instability decays as path diversity increases under limited information.
    """
    print("--- Generating Figure 13 (Envelope Plot) ---")

    # P_values going higher to emphasize the tail
    P_values = [2, 5, 10, 20, 40, 60, 80, 100]
    seeds_per_p = 5
    K_SUBSET = 5 

    results_spread = []
    results_fairness = []

    print(f"Simulating with Sub-selection (k={K_SUBSET})...")

    for P in P_values:
        s_runs = []
        f_runs = []
        for _ in range(seeds_per_p):
            s, f = run_simulation(P, rho=0.2, sigma=0.5, k_subset=K_SUBSET) 
            s_runs.append(s)
            f_runs.append(f)
        
        avg_s = np.mean(s_runs)
        avg_f = np.mean(f_runs)
        results_spread.append(avg_s)
        results_fairness.append(avg_f)
        print(f"  P={P}: Amplitude={avg_s:.3f}")

    # --- PLOTTING ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Red Line (Amplitude)
    color_amp = '#D62728' # tab:red
    ax1.set_xlabel('Number of Paths ($P$)', fontsize=14)
    ax1.set_ylabel('Normalized Oscillation Amplitude\n$(Delta Load / Capacity)$', color=color_amp, fontsize=14)
    ax1.plot(P_values, results_spread, color=color_amp, marker='o', linewidth=2, label='Oscillation Amplitude')
    ax1.tick_params(axis='y', labelcolor=color_amp)
    ax1.grid(True, alpha=0.3)

    # Blue Line (Fairness)
    ax2 = ax1.twinx()  
    color_fair = '#1F77B4' # tab:blue
    ax2.set_ylabel(r'Fairness Metric ($\eta$)', color=color_fair, fontsize=14)
    ax2.plot(P_values, results_fairness, color=color_fair, marker='s', linestyle='--', linewidth=2, label='Fairness Metric')
    ax2.tick_params(axis='y', labelcolor=color_fair)

    # Dynamic placement for annotations based on data
    y_lim = ax1.get_ylim()[1]
    ax1.text(10, y_lim*0.9, "Synchronized\nRegime", ha='center', fontsize=10, style='italic', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax1.text(80, y_lim*0.9, "De-synchronized\nRegime", ha='center', fontsize=10, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig13_envelope_plot')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 13 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_13()