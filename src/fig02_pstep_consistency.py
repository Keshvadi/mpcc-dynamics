import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

# Suppress runtime warnings that can occur during numerical simulation (e.g., division by zero)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Helper Functions ---

def alpha_S_func(tau):
    """
    Accelerating additive increase (linear).
    alpha(tau) = 2*tau if tau < 5 else 1
    """
    return np.where(tau < 5, 2 * tau, 1)

def alpha_1_func(tau):
    """Constant additive increase. alpha(tau) = 1"""
    return 1

def calculate_alpha_hat_p(alpha_func, rho, P, num_terms=30):
    """Calculates the expected additive increase per path rank."""
    alpha_hat = np.zeros(P)
    term_const = 1 - (1 - rho)**(P - 1)
    
    # Fallback for rho=0 case
    if term_const == 0: return np.ones(P) * alpha_func(0)
    
    for p_rank in range(P):
        series_sum = sum(
            (1 - rho)**(k * (P - 1)) * alpha_func(P * k + p_rank)
            for k in range(num_terms)
        )
        alpha_hat[p_rank] = term_const * series_sum
    return alpha_hat

def calculate_f_hat_p_vector(rho, sigma, P, alpha_func, N=1):
    """
    Calculates the equilibrium load vector f^(p) to check for consistency.
    Derived from the fixed-point equations.
    """
    if rho == 0 or rho == 1: return np.zeros(P) # Avoid edge case instability
    
    a_hat_p = np.array([((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)])
    alpha_hat_p = calculate_alpha_hat_p(alpha_func, rho, P)
    
    z_rho_p = (1 - (1 - rho)**(P - 1)) / (rho * (1 - rho)**(P - 1))
    q_factor = 1 + rho * sigma * z_rho_p
    
    common_ratio = q_factor * (1 - rho)**(P - 1)
    denominator = 1 - common_ratio
    
    if abs(denominator) < 1e-9: # Divergent case
        return np.full(P, np.inf)

    f_hat_p = np.zeros(P)
    for p_rank in range(P):
        t1 = sum((1-rho)**(p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank))
        t2 = (1-rho)**p_rank * alpha_hat_p[P-1] * a_hat_p[P-1]
        t3 = q_factor * sum((1-rho)**(P-1+p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank, P-1))
        f_hat_p[p_rank] = (t1 + t2 + t3) / denominator
    return f_hat_p

def is_inconsistent(rho, sigma, P, alpha_func):
    """
    Checks if a parameter set (rho, sigma) violates the P-step oscillation assumption.
    Returns True if the rank ordering is violated (i.e., Load[p] < Load[p+1]).
    """
    f_hat_p = calculate_f_hat_p_vector(rho, sigma, P, alpha_func)
    
    # Divergence is not considered an inconsistency in rank ordering
    if np.any(np.isinf(f_hat_p)): return False 
    
    for p in range(P - 1):
        if f_hat_p[p] < f_hat_p[p+1]:
            return True # Found an inconsistency
    return False

def plot_figure_2():
    """
    Generates Figure 2: Logical consistency of the P-step oscillation.
    """
    
    # --- Configuration ---
    alpha_configs = {
        'alpha_1': {
            'func': alpha_1_func,
            'title': r'Constant Increase'
        },
        'alpha_S': {
            'func': alpha_S_func,
            'title': r'Accelerating Increase'
        }
    }
    P_vals = [2, 3, 4, 5, 6, 7]
    
    # Grid resolution for heatmap
    grid_res = 50
    rho_vals = np.linspace(0.01, 0.99, grid_res)
    sigma_vals = np.linspace(0.0, 1.0, grid_res)
    Rho, Sigma = np.meshgrid(rho_vals, sigma_vals)

    print("Calculating consistency regions... (this may take a moment)")
    
    fig, axes = plt.subplots(2, 6, figsize=(12, 7), sharex=True, sharey=True)
    
    # --- Global Labels ---
    fig.supxlabel(r"Responsiveness ($\rho$)", fontsize=16, y=0.07)
    fig.supylabel(r"Reset Softness ($\sigma$)", fontsize=16, x=0.07)

    # --- Main Plotting Loop ---
    for row, config in enumerate(alpha_configs.values()):
        alpha_func = config['func']
        row_title = config['title']
        
        for col, P in enumerate(P_vals):
            ax = axes[row, col]
            
            # Compute inconsistency boolean map
            Z = np.zeros_like(Rho, dtype=bool)
            for i in range(grid_res):
                for j in range(grid_res):
                    rho_pt, sigma_pt = Rho[i, j], Sigma[i, j]
                    Z[i, j] = is_inconsistent(rho_pt, sigma_pt, P, alpha_func)
            
            # Plot shaded region (Red = Inconsistent)
            ax.contourf(Rho, Sigma, Z, levels=[0.5, 1.5], colors=['#d6616b'])
            
            # Annotate percentage of space that is inconsistent
            percentage = np.mean(Z) * 100
            ax.text(0.5, 0.5, f'{percentage:.1f}%', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, color='white',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#d6616b', ec='none'))
            
            # Column Titles (Number of Paths)
            if row == 0:
                ax.set_title(f"P = {P}", fontsize=16)
            
            # Row Titles (Growth Function)
            if col == 0:
                ax.text(-0.35, 0.5, row_title, 
                        ha='center', va='center', 
                        rotation=90,
                        transform=ax.transAxes, 
                        fontsize=14)

    plt.tight_layout() 
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig02_pstep_consistency_plot')
    
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"Figure 2 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_2()