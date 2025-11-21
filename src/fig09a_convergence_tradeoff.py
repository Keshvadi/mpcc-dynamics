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
    """Slow-start-style additive increase (accelerating)."""
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
    """Calculates equilibrium state variables for a given parameter set."""
    # 1. Agent Equilibrium
    a_hat_p = np.array([((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)])
    
    # 2. Growth Equilibrium
    alpha_hat_p = calculate_alpha_hat_p(alpha_func, rho, P)
    
    # 3. Extrapolation Factor
    z_rho_p = (1 - (1 - rho)**(P - 1)) / (rho * (1 - rho)**(P - 1)) if rho > 0 else float('inf')
    q_factor = 1 + rho * sigma * z_rho_p
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        common_ratio = q_factor * (1 - rho)**(P - 1)
        denominator = 1 - common_ratio

    # 4. Flow Equilibrium (Eq. 7 & 8)
    f_hat_p = np.zeros(P)
    if denominator <= 0 or abs(denominator) < 1e-9:
        f_hat_p.fill(float('inf'))
    else:
        for p_rank in range(P):
            t1 = sum((1-rho)**(p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank))
            t2 = (1-rho)**p_rank * alpha_hat_p[P-1] * a_hat_p[P-1]
            t3 = q_factor * sum((1-rho)**(P-1+p_rank-p_prime) * alpha_hat_p[p_prime] * a_hat_p[p_prime] for p_prime in range(p_rank, P-1))
            f_hat_p[p_rank] = (t1 + t2 + t3) / denominator
    
    eq_type = 'lossless' if f_hat_p[0] <= C_pi else 'lossy'
    
    return {
        "eq_type": eq_type, 
        "f_hat_p": f_hat_p, 
        "a_hat_p": a_hat_p, 
        "alpha_hat_p": alpha_hat_p, 
        "common_ratio": common_ratio
    }

def calculate_metrics(ctx, rho, sigma, P, N, C_pi, gamma, alpha_func):
    """Calculates Change in Convergence (Delta Gamma) relative to baseline."""
    C = P * C_pi
    alpha_max = np.max([alpha_func(tau) for tau in range(10)])
    baseline_gamma = (gamma * C) / (C + alpha_max * N)

    # Calculate Gamma_MPCC
    if ctx['eq_type'] == 'lossless':
        if ctx['f_hat_p'][0] > 0:
            gamma_mpcc = ctx['f_hat_p'][-1] / ctx['f_hat_p'][0]
        else:
            gamma_mpcc = 0
    else: # Lossy Regime
        sum_partial = sum(ctx['alpha_hat_p'][:-1])
        sum_all = sum_partial + ctx['alpha_hat_p'][-1]
        a_last = ctx['a_hat_p'][-1]
        
        if abs(sigma - 1.0) > 1e-6:  # sigma != 1
            lambda_mpcc = ctx['common_ratio'] - 1 + (ctx['common_ratio'] * sum_partial + ctx['alpha_hat_p'][-1]) * a_last / C_pi
        else:  # sigma == 1
            lambda_mpcc = (1 - rho)**(1 - P) * sum_all * a_last / C_pi
            
        gamma_mpcc = gamma * (1 - rho)**(P-1) / (lambda_mpcc + 1)
    
    delta_gamma = gamma_mpcc - baseline_gamma
    return delta_gamma

def plot_figure_9a():
    """
    Generates Figure 9a: Change in Convergence vs. Responsiveness.
    Visualizes the non-monotonic relationship between migration speed and stability.
    """
    print("--- Generating Figure 9a (Convergence Trade-off) ---")
    
    P, gamma, N, C_pi = 3, 0.7, 1000, 12000
    alpha_funcs = {r'$\alpha_1$': alpha_1_func, r'$\alpha_S$': alpha_S_func}
    
    rho_vals = np.linspace(0.02, 1.0, 40)
    sigma_vals = np.linspace(0.0, 1.0, 20)

    print("Calculating trade-off surface... (this may take a minute)")
    results = {}
    
    for name, alpha_func in alpha_funcs.items():
        results[name] = {'lossless': {'dg':[]}, 'lossy': {'dg':[]}}
        
        for i, rho in enumerate(rho_vals):
            dg_lossless, dg_lossy = [], []
            for sigma in sigma_vals:
                ctx = get_equilibrium_context(rho, sigma, P, N, C_pi, alpha_func)
                dg = calculate_metrics(ctx, rho, sigma, P, N, C_pi, gamma, alpha_func)
                
                if ctx['eq_type'] == 'lossless':
                    dg_lossless.append(dg)
                else:
                    dg_lossy.append(dg)
            
            # Store min/max for the fill_between plot
            for cat, dg_list in [('lossless', dg_lossless), ('lossy', dg_lossy)]:
                if dg_list:
                    results[name][cat]['dg'].append((np.min(dg_list), np.max(dg_list)))
                else:
                    results[name][cat]['dg'].append((np.nan, np.nan))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'lossless': {'a1': 'purple', 'aS': 'orangered'}, 
              'lossy': {'a1': 'darkcyan', 'aS': 'green'}}

    for name, data in results.items():
        key = 'a1' if '1' in name else 'aS'
        
        min_dg, max_dg = np.array(data['lossless']['dg']).T
        min_de, max_de = np.array(data['lossy']['dg']).T
        
        ax.fill_between(rho_vals, min_dg, max_dg, color=colors['lossless'][key], 
                        alpha=0.3, label=f'Lossless, {name}')
        
        ax.fill_between(rho_vals, min_de, max_de, color=colors['lossy'][key], 
                        alpha=0.5, hatch='..', edgecolor='gray', label=f'Lossy, {name}')

    ax.axhline(0, color='black', linestyle=':', lw=1, label='Static Baseline')
    
    ax.set_ylabel(r'Change in Convergence ($\Delta\gamma$)', fontsize=14)
    ax.set_xlabel(r'Responsiveness ($\rho$)', fontsize=14)
    
    ax.legend(loc='upper right', fontsize=11)
    
    # --- Save ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig09a_convergence_tradeoff')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 9a saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_9a()