import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

def simulate_variance(rho, sigma, P, alpha_func, N, T_max):
    """
    Simulates the Markov process for agent window evolution and computes 
    the population-wide variance (Fairness metric eta).
    """
    # Initialize state: w (window size), tau (continuity time)
    w = np.zeros(N)
    tau = np.zeros(N, dtype=int)
    
    variance_history = np.zeros(T_max)
    
    for t in range(T_max):
        # 1. Record Fairness Metric (Variance of w)
        variance_history[t] = np.var(w)
        
        # 2. Determine Migration
        # Condition: migration only allowed if tau mod P != P-1
        can_migrate = (tau % P) != (P - 1)
        random_triggers = np.random.rand(N)
        
        will_migrate = can_migrate & (random_triggers < rho)
        will_stay = ~will_migrate
        
        # 3. Update State (Eq. 2 & Markov Process)
        next_w = np.zeros(N)
        next_tau = np.zeros(N, dtype=int)

        # Migrating agents: w -> w * sigma, tau -> 0
        next_w[will_migrate] = w[will_migrate] * sigma
        next_tau[will_migrate] = 0
        
        # Incumbent agents: w -> w + alpha(tau), tau -> tau + 1
        next_w[will_stay] = w[will_stay] + alpha_func(tau[will_stay])
        next_tau[will_stay] = tau[will_stay] + 1
        
        # Commit updates
        w = next_w
        tau = next_tau
        
    return variance_history

def plot_figure_7():
    """
    Generates Figure 7: Fairness Metric (Variance) vs. Time.
    Demonstrates how responsiveness (rho) and reset softness (sigma) impact equity.
    """
    print("--- Generating Figure 7 (Fairness/Variance) ---")
    
    # --- Parameters ---
    P = 3
    N = 10000       # Number of agents (samples)
    T_MAX = 41      # Simulation steps
    
    def alpha_func(tau): return 1.0

    # Scenarios from the paper
    scenarios = [
        {'rho': 0.5, 'sigma': 0.5, 'label': r'$\rho=0.5, \sigma=0.5$'},
        {'rho': 0.5, 'sigma': 0.3, 'label': r'$\rho=0.5, \sigma=0.3$'},
        {'rho': 0.4, 'sigma': 0.5, 'label': r'$\rho=0.4, \sigma=0.5$'},
    ]

    # --- Simulation ---
    results = []
    for params in scenarios:
        print(f"  Simulating: {params['label']}...")
        var_trace = simulate_variance(
            rho=params['rho'],
            sigma=params['sigma'],
            P=P,
            alpha_func=alpha_func,
            N=N,
            T_max=T_MAX
        )
        results.append(var_trace)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    time_axis = np.arange(T_MAX)
    
    colors = ['#0077BB', 'gray', '#EE3377'] 
    markers = ['o', 's', '^']

    for i, (params, var_data) in enumerate(zip(scenarios, results)):
        ax.plot(time_axis, var_data, marker=markers[i], color=colors[i], 
                label=params['label'], markersize=6, markevery=2, linewidth=2)

    # Labels using paper notation
    ax.set_xlabel('Time Step ($t$)', fontsize=14)
    ax.set_ylabel(r'Fairness Metric ($\eta = \mathrm{Var}[w_i(t)])$', fontsize=14)
    
    ax.set_xlim(0, 40)
    ax.set_ylim(bottom=0)
    
    ax.grid(linestyle=':', alpha=0.6)
    ax.legend(title=r"Parameters ($\rho, \sigma$)", loc='upper left', fontsize=12, frameon=True)
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig07_fairness_variance')

    # Use tight layout with crop
    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"Figure 7 saved to: {base_filename}.pdf")

    plt.show()

if __name__ == "__main__":
    plot_figure_7()