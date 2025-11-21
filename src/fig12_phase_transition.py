import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("white") # Heatmaps often look better without gridlines

def simulate_stochastic_dynamics(N, P, steps, rho, sigma, C_total, gamma=0.5, k_subset=None):
    """
    Simulates the stochastic agent-based model to capture de-synchronization effects.
    """
    if k_subset is None:
        k_subset = P
        
    # Derived parameters
    C_path = C_total / P
    
    # Initialization
    agents_path = np.random.randint(0, P, size=N)
    agents_window = np.ones(N) * (C_total / N) * 0.5 
    agents_tau = np.zeros(N)
    
    utilization_history = np.zeros((P, steps))
    
    for t in range(steps):
        # 1. Calculate Load per Path
        path_loads = np.bincount(agents_path, weights=agents_window, minlength=P)
        utilization_history[:, t] = path_loads / C_path
        
        # Add noise for tie-breaking
        noisy_loads = path_loads + np.random.normal(0, 1e-3, P)
        
        # 2. Identify Targets (TRUE Probing Logic)
        
        # Vectorized Probing: Each agent samples k paths
        k_curr = min(k_subset, P)
        
        if k_curr == P:
            # Global Knowledge (Original logic): Every agent targets the global minimum
            best_path_index = np.argmin(noisy_loads)
            specific_best_path = np.full(N, best_path_index, dtype=int)
            
            # Migration Condition: Migrate if current path is NOT the best path
            is_better_path_found = (agents_path != best_path_index)
            
        else:
            # Local Probing: Each agent samples k paths independently
            # Generate random probe indices for all agents: Shape (N, k_curr)
            probes = np.random.randint(0, P, size=(N, k_curr))
            
            # Look up the noisy loads for these probes
            probe_loads = noisy_loads[probes]
            
            # Each agent identifies the best path *within their specific probe set*
            local_best_indices = np.argmin(probe_loads, axis=1)
            
            # Map back to the global Path ID (0..P)
            specific_best_path = probes[np.arange(N), local_best_indices]
            
            # Migration Condition: Found path is strictly better than current path
            current_path_loads = noisy_loads[agents_path]
            found_path_loads = noisy_loads[specific_best_path]
            is_better_path_found = found_path_loads < current_path_loads


        # 3. Migration Logic
        # Apply Responsiveness (rho)
        migration_decisions = np.random.random(N) < rho
        
        # Combined condition
        migrating_agents = is_better_path_found & migration_decisions
        
        # Apply Migration
        agents_window[migrating_agents] *= sigma
        agents_tau[migrating_agents] = 0
        agents_path[migrating_agents] = specific_best_path[migrating_agents]
        
        # 4. Congestion Control (Loss and Increase logic remains unchanged)
        current_path_loads = np.bincount(agents_path, weights=agents_window, minlength=P)
        overloaded_paths = np.where(current_path_loads > C_path)[0]
        
        # Fast boolean lookup for overloaded status
        is_overloaded_path = np.zeros(P, dtype=bool)
        is_overloaded_path[overloaded_paths] = True
        agents_on_overloaded = is_overloaded_path[agents_path]
        
        # Multiplicative Decrease (Loss)
        agents_window[agents_on_overloaded] *= gamma
        agents_tau[agents_on_overloaded] = 0
        
        # Additive Increase (approx. alpha=1 for stochastic model)
        agents_window[~agents_on_overloaded] += 1.0
        agents_tau[~agents_on_overloaded] += 1
        
    return utilization_history

def plot_figure_12():
    """
    Generates Figure 12: Phase Transition Heatmaps.
    Visualizes the shift from coherent oscillation (Low P) to noise (High P).
    """
    print("--- Generating Figure 12 (Phase Transition Heatmaps) ---")

    # Parameters
    N_AGENTS = 2000
    STEPS = 150
    RHO = 0.3      
    SIGMA = 0.5
    TOTAL_CAP = 50000
    GAMMA = 0.5
    
    # NEW PARAMETER: Limited Probing (k=5)
    K_PROBE = 5 

    print(f"  Running Low-Diversity Simulation (P=5, k={K_PROBE})...")
    util_low_p = simulate_stochastic_dynamics(
        N=N_AGENTS, P=5, steps=STEPS, rho=RHO, sigma=SIGMA, 
        C_total=TOTAL_CAP, gamma=GAMMA, k_subset=K_PROBE
    )

    print(f"  Running High-Diversity Simulation (P=50, k={K_PROBE})...")
    util_high_p = simulate_stochastic_dynamics(
        N=N_AGENTS, P=50, steps=STEPS, rho=RHO, sigma=SIGMA, 
        C_total=TOTAL_CAP, gamma=GAMMA, k_subset=K_PROBE
    )

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    cmap = 'inferno' 
    vmin, vmax = 0.0, 1.5 

    # Plot 1: Low P
    im1 = axes[0].imshow(util_low_p, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[0].set_title(f"Low Path Diversity ($P=5$, $k={K_PROBE}$): Coherent Oscillation", fontsize=14)
    axes[0].set_ylabel("Path Index", fontsize=12)
    
    # Only show x-labels on bottom plot
    axes[0].set_xticklabels([])

    # Plot 2: High P
    im2 = axes[1].imshow(util_high_p, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    axes[1].set_title(f"High Path Diversity ($P=50$, $k={K_PROBE}$): De-synchronization (Fractured Herd)", fontsize=14)
    axes[1].set_ylabel("Path Index", fontsize=12)
    axes[1].set_xlabel("Time Step ($t$)", fontsize=12)

    # Shared Colorbar
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.04)
    cbar.set_label(r'Path Utilization ($\hat{L}_{\pi} / C_{\pi}$)', fontsize=12)
    # Mark capacity line
    cbar.ax.plot([0, 1], [1.0, 1.0], 'w-', linewidth=2)

    # fig.suptitle(rf"The Phase Transition: From Synchronization to Noise ($N={N_AGENTS}, \rho={RHO}$) - Probing Model", fontsize=16)

    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig12_phase_transition')

    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 12 (Fixed) saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_12()