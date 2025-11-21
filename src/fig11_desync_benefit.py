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
    return np.where(tau < 5, 2**tau, 1.0) 

def alpha_1_func(tau):
    """Constant additive increase."""
    return 1.0

def calculate_alpha_hat_p(alpha_func, rho, P, num_terms=50):
    """Calculates expected additive increase per path rank."""
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        common_ratio = q_factor * (1 - rho)**(P - 1)
        denominator = 1 - common_ratio

    if abs(denominator) < 1e-9:
        f_hat_0 = float('inf')
    else:
        # Calculate flow on rank 0
        t1 = 0 
        t2 = (1 - rho)**0 * alpha_hat_p[P-1] * a_hat_p[P-1]
        t3 = q_factor * sum((1 - rho)**(P-1-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(0, P-1))
        f_hat_0 = (t1 + t2 + t3) / denominator

    eq_type = 'lossless' if f_hat_0 <= C_pi else 'lossy'
    
    return {
        "eq_type": eq_type, 
        "f_hat_0": f_hat_0, 
        "a_hat_p": a_hat_p,
        "alpha_hat_p": alpha_hat_p, 
        "common_ratio": common_ratio
    }

def calculate_delta_lambda(ctx, rho, sigma, P, N, C_pi, alpha_func):
    """
    Calculates Change in Loss Avoidance (Delta Lambda) relative to static baseline.
    Negative values indicate improvement (De-synchronization benefit).
    """
    C = P * C_pi
    
    # Baseline (Static) Loss Rate
    # If static (rho=0), agents sync up. We approximate max load based on max alpha.
    alpha_max = np.max(alpha_S_func(np.arange(10))) if alpha_func == alpha_S_func else 1.0
    baseline_lambda = (alpha_max * N) / C

    # MPCC Loss Rate
    if ctx['eq_type'] == 'lossless':
        mpcc_lambda = 0.0
    else: # Lossy
        sum_alpha_partial = sum(ctx['alpha_hat_p'][:-1])
        sum_alpha = sum_alpha_partial + ctx['alpha_hat_p'][-1]
        a_last = ctx['a_hat_p'][-1]
        
        if abs(sigma - 1.0) > 1e-6:  # sigma != 1
            mpcc_lambda = ctx['common_ratio'] - 1 + (ctx['common_ratio'] * sum_alpha_partial + ctx['alpha_hat_p'][-1]) * a_last / C_pi
        else:  # sigma == 1
            mpcc_lambda = (1 - rho)**(1 - P) * sum_alpha * a_last / C_pi

    return mpcc_lambda - baseline_lambda

def plot_figure_11():
    """
    Generates Figure 11: Change in Loss Avoidance vs. Responsiveness.
    Shows the 'De-synchronization Benefit' where migration reduces loss 
    compared to static routing for bursty traffic.
    """
    print("--- Generating Figure 11 (De-synchronization Benefit) ---")
    
    # --- Parameters ---
    P, gamma, N, C_pi = 3, 0.7, 1000, 12000
    
    alpha_funcs = {
        r'$\alpha_1$': alpha_1_func,
        r'$\alpha_S$': alpha_S_func
    }
    
    rho_vals = np.linspace(0.02, 1.0, 50)
    sigma_vals = np.linspace(0.0, 1.0, 25)

    # --- Main Calculation Loop ---
    print("Calculating data... (this may take a moment)")
    results = {}
    
    for name, alpha_func in alpha_funcs.items():
        results[name] = {'lossless': [], 'lossy': []}
        
        for rho in rho_vals:
            dl_lossless, dl_lossy = [], []
            for sigma in sigma_vals:
                ctx = get_equilibrium_context(rho, sigma, P, N, C_pi, alpha_func)
                dl = calculate_delta_lambda(ctx, rho, sigma, P, N, C_pi, alpha_func)
                
                if ctx['eq_type'] == 'lossless':
                    dl_lossless.append(dl)
                else:
                    dl_lossy.append(dl)
            
            # Store range tuples
            results[name]['lossless'].append((np.min(dl_lossless), np.max(dl_lossless)) if dl_lossless else (np.nan, np.nan))
            results[name]['lossy'].append((np.min(dl_lossy), np.max(dl_lossy)) if dl_lossy else (np.nan, np.nan))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'lossless': {'a1': 'darkblue', 'aS': 'darkred'}, 
              'lossy': {'a1': 'skyblue', 'aS': 'salmon'}}

    for name, data in results.items():
        key = 'a1' if '1' in name else 'aS'
        
        # Plot Lossless Region
        min_dl, max_dl = np.array(data['lossless']).T
        ax.fill_between(rho_vals, min_dl, max_dl, color=colors['lossless'][key], 
                        alpha=0.5, label=f'Lossless, {name}')
        
        # Plot Lossy Region
        min_dl, max_dl = np.array(data['lossy']).T
        ax.fill_between(rho_vals, min_dl, max_dl, color=colors['lossy'][key], 
                        alpha=0.4, label=f'Lossy, {name}')
        
        # Add boundary lines for clarity in Lossy regions
        ax.plot(rho_vals, min_dl, color=colors['lossy'][key], linestyle=':', lw=1.5)
        ax.plot(rho_vals, max_dl, color=colors['lossy'][key], linestyle='--', lw=1.5)

    # --- Styling ---
    ax.axhline(0, color='black', linestyle=':', lw=1, label='Static Baseline')
    
    ax.set_ylabel(r'Change in Loss Avoidance ($\Delta \lambda$)', fontsize=14)
    ax.set_xlabel(r'Responsiveness ($\rho$)', fontsize=14)
    ax.set_xlim(0, 1)
    
    ax.legend(loc='best', fontsize=11)
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig11_desync_benefit')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 11 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_11()