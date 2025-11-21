import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

# --- Helper Functions ---

def alpha_S_func(tau):
    """Slow-start-style additive increase. Handles array inputs."""
    return np.where(tau < 5, 2**tau, 1)

def alpha_1_func(tau):
    """Constant additive increase."""
    return 1

def calculate_alpha_hat_p(alpha_func, rho, P, num_terms=30):
    """Calculates the expected additive increase per path rank."""
    alpha_hat = np.zeros(P)
    term_const = 1 - (1 - rho)**(P - 1)
    
    if term_const == 0: return np.ones(P) * alpha_func(0)
    
    for p_rank in range(P):
        series_sum = sum(
            (1 - rho)**(k * (P - 1)) * alpha_func(P * k + p_rank)
            for k in range(num_terms)
        )
        alpha_hat[p_rank] = term_const * series_sum
    return alpha_hat

def get_equilibrium_context(rho, sigma, P, N, C_pi, alpha_func):
    """
    Calculates equilibrium state to determine if the system is in a 
    Lossless or Lossy regime.
    """
    a_hat_p = np.array([((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)])
    alpha_hat_p = calculate_alpha_hat_p(alpha_func, rho, P)
    z_rho_p = (1 - (1 - rho)**(P - 1)) / (rho * (1 - rho)**(P - 1)) if rho > 0 else float('inf')
    q_factor = 1 + rho * sigma * z_rho_p
    
    # Suppress potential division warnings during intermediate calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        common_ratio = q_factor * (1 - rho)**(P - 1)
        denominator = 1 - common_ratio

    f_hat_p = np.zeros(P)
    if abs(denominator) < 1e-9:
        f_hat_p.fill(float('inf'))
    else:
        for p_rank in range(P):
            t1 = sum((1-rho)**(p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank))
            t2 = (1-rho)**p_rank * alpha_hat_p[P-1] * a_hat_p[P-1]
            t3 = q_factor * sum((1-rho)**(P-1+p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank, P-1))
            f_hat_p[p_rank] = (t1 + t2 + t3) / denominator
    
    eq_type = 'lossless' if f_hat_p[0] <= C_pi else 'lossy'
    return {"eq_type": eq_type}

def simulate_variance(rho, sigma, P, alpha_func, N_samples=2000, T_max=60):
    """
    Nested simulation to find the steady-state variance for fairness (eta).
    """
    cwnds = np.zeros(N_samples)
    taus = np.zeros(N_samples, dtype=int)
    
    for t in range(T_max):
        # Determine migration candidates
        can_migrate = (taus % P) != (P - 1)
        random_triggers = np.random.rand(N_samples)
        
        will_migrate = can_migrate & (random_triggers < rho)
        will_not_migrate = ~will_migrate
        
        next_cwnds = np.zeros(N_samples)
        next_taus = np.zeros(N_samples, dtype=int)
        
        # Apply Dynamics
        next_cwnds[will_migrate] = cwnds[will_migrate] * sigma
        next_taus[will_migrate] = 0
        
        next_cwnds[will_not_migrate] = cwnds[will_not_migrate] + alpha_func(taus[will_not_migrate])
        next_taus[will_not_migrate] = taus[will_not_migrate] + 1
        
        cwnds, taus = next_cwnds, next_taus
        
    return np.var(cwnds)

def calculate_metrics(rho, sigma, P, alpha_func):
    """Calculates Fairness (eta) metric via simulation."""
    eta_mpcc = simulate_variance(rho, sigma, P, alpha_func)
    return eta_mpcc

def plot_figure_9b():
    """
    Generates Figure 9b: Change in Fairness vs. Responsiveness.
    """
    print("--- Generating Figure 9b (Fairness Trade-off) ---")
    
    P, gamma, N, C_pi = 3, 0.7, 1000, 12000
    alpha_funcs = {r'$\alpha_1$': alpha_1_func, r'$\alpha_S$': alpha_S_func}
    
    rho_vals = np.linspace(0.02, 1.0, 40)
    sigma_vals = np.linspace(0.0, 1.0, 20)

    print("Calculating data for Fairness plot... (this will take several minutes)")
    results = {}
    
    for name, alpha_func in alpha_funcs.items():
        # Store results for Lossless and Lossy regimes separately
        results[name] = {'lossless': {'eta':[]}, 'lossy': {'eta':[]}} 
        
        for i, rho in enumerate(rho_vals):
            print(f"  Processing rho = {rho:.2f} ({i+1}/{len(rho_vals)})...")
            eta_lossless, eta_lossy = [], []
            
            for sigma in sigma_vals:
                # Check regime type
                ctx = get_equilibrium_context(rho, sigma, P, N, C_pi, alpha_func)
                # Calculate metric
                eta = calculate_metrics(rho, sigma, P, alpha_func)
                
                if ctx['eq_type'] == 'lossless':
                    eta_lossless.append(eta)
                else:
                    eta_lossy.append(eta)
            
            # Store min/max range for the shaded plot areas
            for cat, eta_list in [('lossless', eta_lossless), ('lossy', eta_lossy)]:
                if eta_list:
                    results[name][cat]['eta'].append((np.min(eta_list), np.max(eta_list)))
                else:
                    results[name][cat]['eta'].append((np.nan, np.nan))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'lossless': {'a1': 'purple', 'aS': 'orangered'}, 
              'lossy': {'a1': 'darkcyan', 'aS': 'green'}}

    for name, data in results.items():
        key = 'a1' if '1' in name else 'aS'
        
        # Plot Lossless Areas
        min_eta, max_eta = np.array(data['lossless']['eta']).T
        ax.fill_between(rho_vals, min_eta, max_eta, color=colors['lossless'][key], 
                        alpha=0.3, label=f'Lossless, {name}')
        
        # Plot Lossy Areas
        min_eta, max_eta = np.array(data['lossy']['eta']).T
        ax.fill_between(rho_vals, min_eta, max_eta, color=colors['lossy'][key], 
                        alpha=0.5, hatch='..', edgecolor='gray', label=f'Lossy, {name}')

    # Formatting
    ax.set_ylabel(r'Fairness ($\eta$) [log scale]', fontsize=14)
    ax.set_xlabel(r'Responsiveness ($\rho$)', fontsize=14)
    ax.set_yscale('log')
    
    ax.legend(loc='upper right', fontsize=11)
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig09b_fairness_tradeoff')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 9b saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_9b()