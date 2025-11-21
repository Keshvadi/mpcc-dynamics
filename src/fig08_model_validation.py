import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

def run_deterministic_model(rho, sigma, P, N, C_pi, alpha, gamma, T_max):
    """
    Runs the deterministic "Expected MPCC Dynamics" model (Eq. 3).
    Returns time steps, normalized agent counts, and normalized path loads.
    """
    t_steps = np.arange(T_max)

    # State variables: Expected agents (a_hat) and Load (L_hat)
    a_hat = np.zeros((P, T_max)) 
    L_hat = np.zeros((P, T_max)) 

    # Initial conditions
    a_hat[:, 0] = N / P             
    L_hat[:, 0] = 0.4 * C_pi        

    # --- Main Simulation Loop ---
    for t in range(T_max - 1):
        # Identify target path (pi_min)
        pi_min_idx = np.argmin(L_hat[:, t])
        
        for pi in range(P):
            current_a = a_hat[pi, t]
            current_L = L_hat[pi, t]
            
            # --- Update Agent Dynamics (Eq. 3a)
            if pi == pi_min_idx:
                a_hat[pi, t+1] = (1 - rho) * current_a + rho * N
            else:
                a_hat[pi, t+1] = (1 - rho) * current_a
                
            # --- Update Load Dynamics (Eq. 3b)
            z = (N - current_a) / current_a if current_a > 0 else 0
            is_pi_min = (pi == pi_min_idx)
            is_over_capacity = (current_L > C_pi)
            
            if not is_over_capacity:
                if is_pi_min:
                    L_hat[pi, t+1] = (1 + rho * sigma * z) * current_L + current_a * alpha
                else:
                    L_hat[pi, t+1] = (1 - rho) * current_L + (1 - rho) * current_a * alpha
            else: # Over capacity (Congestion Event)
                if is_pi_min:
                    L_hat[pi, t+1] = (gamma + rho * sigma * z) * current_L
                else:
                    L_hat[pi, t+1] = gamma * (1 - rho) * current_L

    # Normalize for plotting
    normalized_a = a_hat / N
    normalized_L = L_hat / C_pi
    
    return t_steps, normalized_a, normalized_L

def plot_figure_8():
    """
    Generates Figure 8: Validation of the expected dynamics model.
    Compares the deterministic model across three stability regimes.
    """
    print("--- Generating Figure 8 (Model Validation) ---")

    # --- Parameters ---
    P = 3          
    N = 1000       
    C_pi = 12000   
    gamma = 0.7    # Multiplicative Decrease (beta -> gamma)
    alpha = 1.0    
    T_max = 51     
    
    # Regimes defined in the paper
    param_sets = [
        {'rho': 0.1, 'sigma': 0.50},  # (a) Lossless Regime
        {'rho': 0.05, 'sigma': 0.50}, # (b) Lossy Regime
        {'rho': 0.1, 'sigma': 1.00}   # (c) High-Loss Regime
    ]
    
    titles = [
        "(a) Lossless Regime\n" + r"($\rho=0.1, \sigma=0.5$)",
        "(b) Lossy Regime\n" + r"($\rho=0.05, \sigma=0.5$)",
        "(c) High-Loss Regime\n" + r"($\rho=0.1, \sigma=1.0$)"
    ]

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    agent_color = 'm'         
    load_color = 'tab:orange'   
    linestyles = [':', '--', '-'] # Path 0, 1, 2
    line_width = 2.0            

    for i, params in enumerate(param_sets):
        ax = axes[i]
        
        t_steps, norm_a, norm_L = run_deterministic_model(
            rho=params['rho'], sigma=params['sigma'],
            P=P, N=N, C_pi=C_pi, alpha=alpha, gamma=gamma, T_max=T_max
        )
        
        ax.set_title(titles[i], fontsize=14)
        
        # Capacity line
        ax.axhline(1.0, color='grey', linestyle='--')
        
        # Plot dynamics for all paths
        for pi in range(P):
            ls = linestyles[pi]
            ax.plot(t_steps, norm_a[pi, :], color=agent_color, linestyle=ls, linewidth=line_width)
            ax.plot(t_steps, norm_L[pi, :], color=load_color, linestyle=ls, linewidth=line_width)

        ax.set_xlabel('Time Step ($t$)', fontsize=14)
        ax.set_xlim(0, T_max - 1)
    
    axes[0].set_ylabel(r'Normalized Agents ($\hat{a}_{\pi}/N$) & Path Utilization', fontsize=14)
    axes[0].set_ylim(0.0, 1.1)

    # --- Global Legend ---
    # Proxy artists
    l_a0 = plt.Line2D([0], [0], color=agent_color, ls=linestyles[0], lw=line_width)
    l_a1 = plt.Line2D([0], [0], color=agent_color, ls=linestyles[1], lw=line_width)
    l_a2 = plt.Line2D([0], [0], color=agent_color, ls=linestyles[2], lw=line_width)
    l_u0 = plt.Line2D([0], [0], color=load_color, ls=linestyles[0], lw=line_width)
    l_u1 = plt.Line2D([0], [0], color=load_color, ls=linestyles[1], lw=line_width)
    l_u2 = plt.Line2D([0], [0], color=load_color, ls=linestyles[2], lw=line_width)

    handles = [l_a0, l_a1, l_a2, l_u0, l_u1, l_u2]
    labels = [
        r'$\hat{a}_0(t)/N$', r'$\hat{a}_1(t)/N$', r'$\hat{a}_2(t)/N$',
        r'$\hat{L}_0(t)/C_{\pi}$', r'$\hat{L}_1(t)/C_{\pi}$', r'$\hat{L}_2(t)/C_{\pi}$'
    ]
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)
    
    # --- Save Output ---
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig08_model_validation')
    
    plt.tight_layout()
    # bbox_inches='tight' is crucial here to include the external legend
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight')
    print(f"Figure 8 saved to: {base_filename}.pdf")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_8()