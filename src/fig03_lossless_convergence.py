import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Style Setup ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

def plot_figure_3a():
    """
    Generates Figure 3-a: Convergence of Expected Agent Count.
    Simulation with rho=0.1 showing exponential convergence to equilibrium.
    """
    print("--- Generating Figure 3-a (Agent Count) ---")
    
    # --- Parameters ---
    P = 3           # Number of paths
    rho = 0.1       # Responsiveness (rho)
    N = 1000        # Total number of agents
    T_MAX = 41      # Simulation time steps

    # --- 1. Calculate Equilibrium Agent Levels (Eq. 6) ---
    # a_hat^(p) formula
    equilibrium_levels = np.array([
        ((1 - rho)**p * rho * N) / (1 - (1 - rho)**P) for p in range(P)
    ])
    a_eq_0, a_eq_1, a_eq_2 = equilibrium_levels
    
    print(f"  Equilibrium Levels: a^(0)={a_eq_0:.2f}, a^(1)={a_eq_1:.2f}, a^(2)={a_eq_2:.2f}")

    # --- 2. Simulate Agent Dynamics (Eq. 3a) ---
    agents_history = np.zeros((T_MAX, P))
    agents_history[0, :] = N / P
    
    for t in range(T_MAX - 1):
        current_agents = agents_history[t, :]
        next_agents = np.zeros(P)
        # Determine which path is the target (cyclic behavior in this deterministic model)
        pi_min_idx = (P - 1 - t) % P
        
        for i in range(P):
            if i == pi_min_idx:
                next_agents[i] = current_agents[i] + rho * (N - current_agents[i])
            else:
                next_agents[i] = (1 - rho) * current_agents[i]
        agents_history[t+1, :] = next_agents

    # --- 3. Calculate Trajectory Functions (Convergence check) ---
    t_values = np.linspace(0, T_MAX - 1, 200)
    a_start = N / P
    trajectory_0 = (a_start - a_eq_0) * (1 - rho)**t_values + a_eq_0
    trajectory_1 = (a_start - a_eq_1) * (1 - rho)**t_values + a_eq_1
    trajectory_2 = (a_start - a_eq_2) * (1 - rho)**t_values + a_eq_2

    # --- 4. Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))
    time_steps = np.arange(T_MAX)

    path_colors = ['#0077BB', '#EE7733', '#EE3377']
    equilibrium_color = 'crimson'
    trajectory_color = 'dimgray'

    # Plot Equilibrium Levels
    ax.axhline(y=a_eq_0, color=equilibrium_color, linestyle='--', label='Equilibrium Levels')
    ax.axhline(y=a_eq_1, color=equilibrium_color, linestyle='--', label='_nolegend_')
    ax.axhline(y=a_eq_2, color=equilibrium_color, linestyle='--', label='_nolegend_')

    # Plot Simulation Dynamics
    ax.plot(time_steps, agents_history[:, 0], '-o', color=path_colors[0], markersize=5, label='Path 1 Agents')
    ax.plot(time_steps, agents_history[:, 1], '-o', color=path_colors[1], markersize=5, label='Path 2 Agents')
    ax.plot(time_steps, agents_history[:, 2], '-o', color=path_colors[2], markersize=5, label='Path 3 Agents')

    # Plot Theoretical Trajectories
    ax.plot(t_values, trajectory_0, '--', color=trajectory_color, label='Theoretical Convergence')
    ax.plot(t_values, trajectory_1, '--', color=trajectory_color, label='_nolegend_')
    ax.plot(t_values, trajectory_2, '--', color=trajectory_color, label='_nolegend_')
    
    # Annotations
    ax.annotate(r'$\hat{a}^{(0)}$', xy=(38, a_eq_0), xytext=(41, a_eq_0 + 20), fontsize=12)
    ax.annotate(r'$\hat{a}^{(1)}$', xy=(38, a_eq_1), xytext=(41, a_eq_1 + 20), fontsize=12)
    ax.annotate(r'$\hat{a}^{(2)}$', xy=(38, a_eq_2), xytext=(41, a_eq_2 - 40), fontsize=12)

    # Formatting
    ax.set_xlabel('Time Step ($t$)', fontsize=14)
    ax.set_ylabel(r'Expected Agent Count ($\hat{a}_{\pi}(t)$)', fontsize=14)
    ax.set_xlim(0, 45) # Extended slightly for annotations
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='best', fontsize=12, frameon=True, shadow=True)

    # Save
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_filename = os.path.join(output_dir, 'fig03-a_agent_convergence')
    plt.tight_layout() 
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"  Saved {base_filename}.pdf")

def plot_figure_3b():
    """
    Generates Figure 3-b: Convergence of Expected Path Load.
    Simulation with rho=0.2 showing load oscillation within equilibrium bounds.
    """
    print("--- Generating Figure 3-b (Path Load) ---")

    # --- 1. Parameters ---
    P = 3           # Number of Paths
    N = 1000        # Total Agents
    rho = 0.2       # Responsiveness (rho)
    sigma = 0.5     # Reset Softness (sigma)
    alpha = 1.0     # Average Additive Increase
    T_max = 251     # Simulation duration

    # --- 2. Analytical Equilibrium (Eq. 7 & 8) ---
    ranks = np.arange(P)
    a_eq = ((1 - rho)**ranks * rho * N) / (1 - (1 - rho)**P)

    # Extrapolation factor z(.)
    z_eq = np.zeros(P)
    for p in range(P):
        if a_eq[p] > 0:
            z_eq[p] = (N - a_eq[p]) / a_eq[p]

    # Solve linear system for Load Equilibrium
    # M * L_eq = V
    M_sys = np.array([
        [1, 0, -(1 + rho * sigma * z_eq[2])],
        [-(1 - rho), 1, 0],
        [0, -(1 - rho), 1]
    ])
    V_sys = np.array([
        alpha * a_eq[2],
        (1 - rho) * alpha * a_eq[0],
        (1 - rho) * alpha * a_eq[1]
    ])
    L_eq = np.linalg.solve(M_sys, V_sys)
    
    print(f"  Path Load Equilibrium: {np.round(L_eq, 2)}")

    # --- 3. Simulation ---
    t_steps = np.arange(T_max)
    a_hat = np.zeros((P, T_max)) 
    L_hat = np.zeros((P, T_max)) 
    
    # Init
    a_hat[:, 0] = N / P
    L_hat[:, 0] = 10.0

    for t in range(T_max - 1):
        # Greedy selection: find path with min load
        pi_min_idx = np.argmin(L_hat[:, t])
        
        for pi in range(P):
            current_a = a_hat[pi, t]
            current_L = L_hat[pi, t]
            
            # Update Agent Dynamics
            if pi == pi_min_idx:
                a_hat[pi, t+1] = (1 - rho) * current_a + rho * N
            else:
                a_hat[pi, t+1] = (1 - rho) * current_a
                
            # Update Path Load Dynamics (Eq 3b)
            if pi == pi_min_idx:
                z_t = (N - current_a) / current_a if current_a > 0 else 0
                L_hat[pi, t+1] = (1 + rho * sigma * z_t) * current_L + alpha * current_a
            else:
                L_hat[pi, t+1] = (1 - rho) * (current_L + alpha * current_a)

    # --- 4. Plotting ---
    path_colors = ['#0077BB', '#EE7733', '#EE3377'] 

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Load
    for pi in range(P):
        ax.plot(t_steps, L_hat[pi, :], color=path_colors[pi], label=f'Path {pi+1} Load')

    # Plot Equilibrium Bounds
    for p in range(P):
        ax.axhline(y=L_eq[p], color='black', linestyle='--', 
                   label='Equilibrium Bounds' if p == 0 else '_nolegend_')

    # Annotations
    ax.annotate(r'$\hat{L}^{(0)}$', xy=(240, L_eq[0]), xytext=(252, L_eq[0]), 
                 fontsize=12, verticalalignment='center')
    ax.annotate(r'$\hat{L}^{(P-1)}$', xy=(240, L_eq[P-1]), xytext=(252, L_eq[P-1]), 
                 fontsize=12, verticalalignment='center')

    # Formatting
    ax.set_xlabel('Time Step ($t$)', fontsize=12)
    ax.set_ylabel(r'Expected Path Load ($\hat{L}_{\pi}(t)$)', fontsize=12)
    ax.set_xlim(0, 265)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower right', fontsize=12)

    # --- Inset: Zoom on Oscillation ---
    ax_inset = ax.inset_axes([0.5, 0.15, 0.35, 0.35])
    for pi in range(P):
        ax_inset.plot(t_steps, L_hat[pi, :], color=path_colors[pi])

    # Inset limits
    inset_xlim = (150, 170)
    inset_mask = (t_steps >= inset_xlim[0]) & (t_steps <= inset_xlim[1])
    inset_ymin = np.min(L_hat[:, inset_mask]) - 50
    inset_ymax = np.max(L_hat[:, inset_mask]) + 50
    
    ax_inset.set_xlim(inset_xlim)
    ax_inset.set_ylim(inset_ymin, inset_ymax)
    ax_inset.set_xlabel('t')
    ax_inset.set_ylabel(r'$\hat{L}_{\pi}(t)$')
    ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    # Save
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    base_filename = os.path.join(output_dir, 'fig03-b_load_convergence')

    plt.tight_layout()
    plt.savefig(f'{base_filename}.png', dpi=300)
    plt.savefig(f'{base_filename}.pdf', dpi=300)
    print(f"  Saved {base_filename}.pdf")

def plot_figure_3_all():
    """Run both sub-figures for Figure 3."""
    plot_figure_3a()
    plot_figure_3b()
    print("\nAll Figure 3 plots generated.")
    plt.show()

if __name__ == "__main__":
    plot_figure_3_all()