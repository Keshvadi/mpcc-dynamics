import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

# --- Helper Functions ---

def alpha_1(tau):
    """Constant additive increase (alpha(tau) = 1)."""
    return 1.0

def alpha_S(tau):
    """Slow-start-style additive increase (alpha(tau) = 2^tau for tau < 5, else 1)."""
    return 2**tau if tau < 5 else 1.0

def calculate_alpha_hat(p_rank, rho, P, alpha_func):
    """
    Calculates the expected additive increase (alpha_hat) for a given path rank.
    Uses geometric series summation for efficiency.
    """
    geo_r = (1 - rho)**(P - 1)
    
    # Handle small rho approximation
    if abs(geo_r - 1) < 1e-10:
        return alpha_func(p_rank)
    
    norm = 1 - geo_r
    
    # Optimization: For constant alpha, the sum is just 1.0
    if alpha_func == alpha_1:
        return 1.0
    
    # For variable alpha (alpha_S), sum the series
    s = 0.0
    r_pow = 1.0
    
    # Find k_max where tau = P * k + p >= 5 (saturation point of alpha_S)
    k_max = int(np.ceil((5 - p_rank) / P))
    if k_max < 0: k_max = 0
        
    for k in range(k_max):
        w = norm * r_pow
        tau = P * k + p_rank
        s += w * alpha_func(tau)
        r_pow *= geo_r
        
    # Add the tail of the series (where alpha_func is always 1)
    s += r_pow * 1.0
    return s

def calculate_equilibrium(rho, sigma, P, N, alpha_func):
    """
    Solves the linear system for equilibrium flows (f_eq).
    Returns (flow_vector, is_valid).
    """
    # Basic validity checks
    if rho <= 0 or rho > 1 or sigma < 0 or sigma > 1:
        return None, False
    
    denom = 1 - (1 - rho)**P
    if abs(denom) < 1e-10:
        return None, False
    
    # Agent Equilibrium
    a_eq = np.array([(1 - rho)**p * rho * N / denom for p in range(P)])
    
    denom_z = rho * (1 - rho)**(P - 1)
    if abs(denom_z) < 1e-10:
        return None, False
    
    # Extrapolation factor
    z = (1 - (1 - rho)**(P - 1)) / denom_z
    
    # Growth Equilibrium
    alpha_hat_vec = np.array([calculate_alpha_hat(p, rho, P, alpha_func) for p in range(P)])
    
    # Set up the linear system A * f = b
    A = np.zeros((P, P))
    b = np.zeros(P)
    
    # Equation for Rank 0 (Jump): f[0] - (1 + rho*sigma*z) * f[P-1] = ...
    A[0, 0] = 1
    A[0, P - 1] = -(1 + rho * sigma * z)
    b[0] = alpha_hat_vec[P - 1] * a_eq[P - 1]
    
    # Equation for Rank p > 0 (Shift): f[p] - (1-rho) * f[p-1] = ...
    for p in range(1, P):
        A[p, p] = 1
        A[p, p - 1] = -(1 - rho)
        b[p] = (1 - rho) * alpha_hat_vec[p - 1] * a_eq[p - 1]
    
    try:
        f_eq = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, False
    
    # Validity check: Strictly decreasing flows (Rank 0 > Rank 1 > ...)
    if not np.all(np.diff(f_eq) < 0):
        return None, False
    
    return f_eq, True

def plot_figure_10():
    """
    Generates Figure 10: Change in Efficiency vs. Responsiveness.
    Compares Lossless equilibria against a Lossy baseline.
    """
    print("--- Generating Figure 10 (Efficiency Trade-off) ---")

    # --- Parameters ---
    P = 3
    N = 1000
    C_total = 36000 # Total Capacity
    gamma = 0.7     # Multiplicative decrease (beta -> gamma)
    
    # Parameter sweep ranges
    rho_vals = np.linspace(0.01, 1.0, 50)
    sigma_vals = np.linspace(0.0, 1.0, 50)
    
    # Scenarios
    alpha_funcs = [(alpha_1, 'α₁'), (alpha_S, 'α_S')]

    # --- Main Calculation Loop ---
    print("Calculating efficiency trade-offs...")
    results = {}
    
    for alpha_func, label in alpha_funcs:
        min_deltas = []
        max_deltas = []
        print(f"  Processing {label}...")
        
        for rho in rho_vals:
            current_deltas = []
            for sigma in sigma_vals:
                f_eq, is_valid = calculate_equilibrium(rho, sigma, P, N, alpha_func)
                
                if is_valid:
                    # Check if in Lossless regime (Max flow <= Path Capacity)
                    # Path Capacity = C_total / P
                    if f_eq[0] <= C_total / P: 
                        # Efficiency (epsilon) = Min Flow / Path Capacity
                        # f_eq[-1] is min flow (Rank P-1)
                        epsilon_mpcc = (f_eq[-1] * P) / C_total
                        delta_epsilon = epsilon_mpcc - gamma
                        current_deltas.append(delta_epsilon)
        
            if current_deltas:
                min_deltas.append(np.min(current_deltas))
                max_deltas.append(np.max(current_deltas))
            else:
                min_deltas.append(np.nan)
                max_deltas.append(np.nan)
                
        results[label] = (min_deltas, max_deltas)
    print("Calculations complete.")

    # --- Theoretical Lossy Baseline ---
    rho_plot_range = np.linspace(0.01, 1.0, 200)
    lossy_epsilon_mpcc = gamma * (1 - rho_plot_range)**(P - 1)
    delta_eps_lossy_line = lossy_epsilon_mpcc - gamma

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot filled regions for lossless cases
    ax.fill_between(rho_vals, results['α₁'][0], results['α₁'][1], 
                    color='purple', alpha=0.5, label=r'Lossless, $\alpha_1$')
    ax.fill_between(rho_vals, results['α_S'][0], results['α_S'][1], 
                    color='orange', alpha=0.5, label=r'Lossless, $\alpha_S$')
    
    # Plot dashed line for lossy regime
    ax.plot(rho_plot_range, delta_eps_lossy_line, color='black', 
            linestyle='--', label='Lossy Regime')
    
    # Plot baseline
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, label='Static Baseline')
    
    # Formatting
    ax.set_ylim([-0.8, 0.3])
    ax.set_xlabel(r'Responsiveness ($\rho$)', fontsize=14)
    ax.set_ylabel(r'Change in Efficiency ($\Delta\epsilon$)', fontsize=14)
    ax.legend(loc='lower left', fontsize=12)

    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.join(output_dir, 'fig10_efficiency')
    
    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 10 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_10()