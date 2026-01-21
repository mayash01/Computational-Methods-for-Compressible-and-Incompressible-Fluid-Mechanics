import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from scipy.sparse import lil_matrix

def idx(i, j, imax): 
    '''
    The pressure Poisson equation is solved as a 1D linear system: Ax=b but the A is a 2D matrix of size imaxXjmax.
    In order to solve it we need to convert the 2D indexs into 1D indexs. p[i,j] --> p[k]
    '''
    return i + j * imax

def boundary_condi(u, v, imax, jmax, u_lid):
    '''
    Lid-driven cavity flow problem: Three walls have no velocity (no-slip condition), and one lid moves with uniform velocity. 
    The lid's velocity is enforced using ghost cells to ensure uniform velocity on the top wall.
    '''
    u[0, :], v[0, :]  = 0.0, -v[1, :]
    u[imax, :], v[imax+1, :] = 0.0, -v[imax, :]
    v[:, 0], u[:, 0] = 0.0, -u[:, 1]
    v[:, jmax], u[:, jmax+1] = 0.0, 2.0 * u_lid - u[:, jmax]

    return u, v

def compute_F(u, v, dt, imax, jmax, dx, dy, Re):
    '''
    Assuming g_x = 0, this function driven straight from the equaions as learned in class
    (Equation 1 lecture 5 / Equation 5 in this report)
    '''
    F = np.zeros_like(u)
    i_s, i_e = 1, imax
    j_s, j_e = 1, jmax + 1
    
    d2u_dx2 = (u[i_s+1:i_e+1, j_s:j_e] - 2*u[i_s:i_e, j_s:j_e] + u[i_s-1:i_e-1, j_s:j_e]) / dx**2
    d2u_dy2 = (u[i_s:i_e, j_s+1:j_e+1] - 2*u[i_s:i_e, j_s:j_e] + u[i_s:i_e, j_s-1:j_e-1]) / dy**2
    
    du2_dx = (((u[i_s:i_e, j_s:j_e] + u[i_s+1:i_e+1, j_s:j_e]) / 2)**2 -
              ((u[i_s-1:i_e-1, j_s:j_e] + u[i_s:i_e, j_s:j_e]) / 2)**2) / dx
    
    duv_dy = (((v[i_s:i_e, j_s:j_e] + v[i_s+1:i_e+1, j_s:j_e]) / 2) *
              ((u[i_s:i_e, j_s:j_e] + u[i_s:i_e, j_s+1:j_e+1]) / 2) -
              ((v[i_s:i_e, j_s-1:j_e-1] + v[i_s+1:i_e+1, j_s-1:j_e-1]) / 2) *
              ((u[i_s:i_e, j_s-1:j_e-1] + u[i_s:i_e, j_s:j_e]) / 2)) / dy
    
    F[i_s:i_e, j_s:j_e] = u[i_s:i_e, j_s:j_e] + dt * ((1/Re) * (d2u_dx2 + d2u_dy2) - du2_dx - duv_dy)
    F[0, :] = u[0, :]
    F[imax, :] = u[imax, :]

    return F

def compute_G(u, v, dt, imax, jmax, dx, dy, Re):
    '''
    Assuming g_y = 0, this function driven straight from the equaions as learned in class
    (Equation 1 lecture 5 / Equation 5 in this report)
    '''
    G = np.zeros_like(v)
    i_s, i_e = 1, imax + 1
    j_s, j_e = 1, jmax
    
    d2v_dx2 = (v[i_s+1:i_e+1, j_s:j_e] - 2*v[i_s:i_e, j_s:j_e] + v[i_s-1:i_e-1, j_s:j_e]) / dx**2
    d2v_dy2 = (v[i_s:i_e, j_s+1:j_e+1] - 2*v[i_s:i_e, j_s:j_e] + v[i_s:i_e, j_s-1:j_e-1]) / dy**2
    
    duv_dx = (((u[i_s:i_e, j_s:j_e] + u[i_s:i_e, j_s+1:j_e+1]) / 2) *
              ((v[i_s:i_e, j_s:j_e] + v[i_s+1:i_e+1, j_s:j_e]) / 2) -
              ((u[i_s-1:i_e-1, j_s:j_e] + u[i_s-1:i_e-1, j_s+1:j_e+1]) / 2) *
              ((v[i_s-1:i_e-1, j_s:j_e] + v[i_s:i_e, j_s:j_e]) / 2)) / dx
    
    dv2_dy = (((v[i_s:i_e, j_s:j_e] + v[i_s:i_e, j_s+1:j_e+1]) / 2)**2 -
              ((v[i_s:i_e, j_s-1:j_e-1] + v[i_s:i_e, j_s:j_e]) / 2)**2) / dy
    
    G[i_s:i_e, j_s:j_e] = v[i_s:i_e, j_s:j_e] + dt * ((1/Re) * (d2v_dx2 + d2v_dy2) - duv_dx - dv2_dy)
    G[:, 0] = v[:, 0]
    G[:, jmax] = v[:, jmax]

    return G

def sparse_matrix(imax, jmax, dx, dy):
    '''
    Coefficient matrix A for the pressure Poisson equation.
    The discrete Poisson equation is modified in cells adjacent to boundaries using boundary flags
    '''
    N = imax * jmax
    dx2, dy2 = dx**2, dy**2
    A = lil_matrix((N, N))
    
    for j in range(jmax):
        for i in range(imax):
            k = idx(i, j, imax)
            if k == 0:
                A[k, k] = 1.0
                continue

            coeff = 0.0
            if i > 0: #West neighbor : if i>0
                A[k, idx(i-1, j, imax)] = 1/dx2
                coeff -= 1/dx2
                
            if i < imax - 1: #East neighbor : if i<imax-1
                A[k, idx(i+1, j, imax)] = 1/dx2
                coeff -= 1/dx2
            
            if j > 0: #South neighbor : if j>0
                A[k, idx(i, j-1, imax)] = 1/dy2
                coeff -= 1/dy2

            if j < jmax - 1: # North neighbor : if j<jmax-1
                A[k, idx(i, j+1, imax)] = 1/dy2
                coeff -= 1/dy2

            A[k, k] = coeff

    return A.tocsr()

def right_side(F, G, dt, imax, jmax, dx, dy):
    '''
     Right-hand side (RHS) vector for the pressure Poisson equation.
     From the continuity equation (incompressibility)
    '''
    b = np.zeros(imax * jmax)
    for j in range(jmax):
        for i in range(imax):
            k = idx(i, j, imax)
            Fx = (F[i+1, j+1] - F[i, j+1]) / dx
            Gy = (G[i+1, j+1] - G[i+1, j]) / dy
            b[k] = (Fx + Gy) / dt
    
    b[0] = 0.0

    return b

def velocities_calculation(u, v, F, G, p_vec, p, dt, imax, jmax, dx, dy):
    '''
    Calculation of NS equations
    '''
    for j in range(jmax):
        for i in range(imax):
            p[i+1, j+1] = p_vec[idx(i, j, imax)]
    p[0, :] = p[1, :]
    p[imax+1, :] = p[imax, :]
    p[:, 0] = p[:, 1]
    p[:, jmax+1] = p[:, jmax]
    
    u[1:imax, 1:jmax+1] = F[1:imax, 1:jmax+1] - dt/dx * (p[2:imax+1, 1:jmax+1] - p[1:imax, 1:jmax+1])
    v[1:imax+1, 1:jmax] = G[1:imax+1, 1:jmax] - dt/dy * (p[1:imax+1, 2:jmax+1] - p[1:imax+1, 1:jmax])

    return u, v, p

def compute_dt(u, v, tau, Re, dx, dy):
    '''
    Stability Condition, tau safety factor is typically chosen between 0 and 1
    '''
    u_max = max(np.max(np.abs(u)), 1e-10)
    v_max = max(np.max(np.abs(v)), 1e-10)
    dt_diff = Re / 2 / (1/dx**2 + 1/dy**2)
    dt_conv_x = dx / u_max
    dt_conv_y = dy / v_max

    return tau * min(dt_diff, dt_conv_x, dt_conv_y) #The criterion can be used to predict the required time step for stability

def convergence(u, v, u_old, v_old):
    du = np.max(np.abs(u - u_old))
    dv = np.max(np.abs(v - v_old))

    return max(du, dv)

def simulation(Re, imax, jmax, t_max=150.0, convergence_tol=1e-7, 
                   save_times=[1.4, 3.4, 5.4, 7.4, 9.4, 12.6]):
    '''
    Run the simulation for Re = 100, 1000. 
    The t_max, i_max and j_max are determined based on the simulation results and
    comparison to the benchmark.
    '''
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/imax, Ly/jmax
    u_lid = 1
    # u_lid = 1.0
    tau = 0.5

    
    print(f"\n{'='*70}")
    print(f"RUNNING SIMULATION FOR Re = {Re}")
    print(f"{'='*70}")
    print(f"Grid: {imax} x {jmax}, dx = {dx:.6f}, dy = {dy:.6f}")
    
    u = np.zeros((imax+2, jmax+2))
    v = np.zeros((imax+2, jmax+2))
    p = np.zeros((imax+2, jmax+2))
    u_old = np.zeros_like(u)
    v_old = np.zeros_like(v)
    
    velocity_snapshots = {}
    save_tolerance = 0.01  # Save if within 0.01 of target time
    
    A_sparse = sparse_matrix(imax, jmax, dx, dy) # Assemble pressure matrix
    
    t = 0.0
    n_iter = 0
    check_interval = 500
     
    u, v = boundary_condi(u, v, imax, jmax, u_lid) # Apply initial BC
    
    converged = False
    while t < t_max and not converged:
        if n_iter % check_interval == 0:
            u_old[:] = u
            v_old[:] = v
        
        dt = compute_dt(u, v, tau, Re, dx, dy)
        if t + dt > t_max:
            dt = t_max - t
        
        F = compute_F(u, v, dt, imax, jmax, dx, dy, Re)
        G = compute_G(u, v, dt, imax, jmax, dx, dy, Re)
        b = right_side(F, G, dt, imax, jmax, dx, dy)
        p_vec = spsolve(A_sparse, b)
        u, v, p = velocities_calculation(u, v, F, G, p_vec, p, dt, imax, jmax, dx, dy)
        u, v = boundary_condi(u, v, imax, jmax, u_lid)
        
        t += dt
        n_iter += 1
        
        for t_target in save_times:
            if abs(t - t_target) < save_tolerance and t_target not in velocity_snapshots:  # Interpolate velocities to cell centers

                u_center = 0.5 * (u[1:imax+1, 1:jmax+1] + u[2:imax+2, 1:jmax+1])
                v_center = 0.5 * (v[1:imax+1, 1:jmax+1] + v[1:imax+1, 2:jmax+2])
                
                velocity_snapshots[t_target] = {
                    'u_center': u_center.copy(),
                    'v_center': v_center.copy(),
                    't_actual': t
                }
                print(f"Saved velocity at t = {t:.4f} (target: {t_target})")
        
        if n_iter % check_interval == 0:
            residual = convergence(u, v, u_old, v_old)
            print(f"Iter {n_iter:6d}, t = {t:8.4f}, residual = {residual:.2e}")
            
            if residual < convergence_tol:
                converged = True
                print(f"{'-'*70}")
                print(f"\n* CONVERGED at iteration {n_iter}, t = {t:.4f} *")
                print(f"{'-'*70}")
                      
    print(f"Simulation complete")

    ghia_y_u = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                         0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
    
    ghia_x_v = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                         0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])
    
    # Centerline locations (always at 0.5 regardless of grid size)
    # x_center = 0.5  
    # y_center = 0.5  
    
    # Convert normalized centerline position to grid index
    i_center = imax//2  # vertical centerline
    j_center = jmax//2  # horizontal centerline
    
    # Extract u-velocity and u-velocity along centerline at Ghia y-coordinates and x-coordinates
    u_sim = []
    for y_norm in ghia_y_u:
        j = int(y_norm * jmax)  # Convert normalized y to grid index
        if 0 <= j <= jmax + 1:
            u_sim.append(u[i_center, j])
    
    v_sim = []
    for x_norm in ghia_x_v:
        i = int(x_norm * imax)  # Convert normalized x to grid index
        if 0 <= i <= imax + 1:
            v_sim.append(v[i, j_center])
    
    return np.array(u_sim), np.array(v_sim), velocity_snapshots

if __name__ == '__main__':
    grid_1000 = 220
    grid_100 = 220
    u_sim_100, v_sim_100, velocity_snapshots = simulation(Re=100.0, imax = grid_100, jmax = grid_100,t_max=50) # Run Re=100
    u_sim_1000, v_sim_1000, velocity_snapshots = simulation(Re=1000.0, imax = grid_1000, jmax = grid_1000, t_max=200) # Run Re=1000

    # Ghia benchmark data for reference
    ghia_y_u = np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531,
                        0.5000, 0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
    ghia_x_v = np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563, 0.2266, 0.2344,
                        0.5000, 0.8047, 0.8594, 0.9063, 0.9453, 0.9531, 0.9609, 0.9688, 1.0000])

    ghia_u_re100 = np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662, -0.21090,
                            -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722, 0.78871, 0.84123, 1.00000])
    ghia_u_re1000 = np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289, -0.27805, -0.10648,
                            -0.06080, 0.05702, 0.18719, 0.33304, 0.46604, 0.51117, 0.57492, 0.65928, 1.00000])

    ghia_v_re100 = np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077, 0.17507, 0.17527,
                            0.05454, -0.24533, -0.22445, -0.16914, -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])
    ghia_v_re1000 = np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095, 0.33075, 0.32235,
                            0.02526, -0.31966, -0.42665, -0.51550, -0.39188, -0.33714, -0.27669, -0.21388, 0.00000])


    cs_u_re100 = CubicSpline(ghia_y_u, ghia_u_re100)
    cs_u_re1000 = CubicSpline(ghia_y_u, ghia_u_re1000)
    
    cs_v_re100 = CubicSpline(ghia_x_v, ghia_v_re100)
    cs_v_re1000 = CubicSpline(ghia_x_v, ghia_v_re1000)

    y_smooth = np.linspace(0, 1, 300)
    x_smooth = np.linspace(0, 1, 300)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    # V-velocity plot
    ax1.plot(x_smooth, cs_v_re100(x_smooth), '-', label='center line v-velocity Re=100', 
            linewidth=1.2, color='lightblue')
    ax1.plot(x_smooth, cs_v_re1000(x_smooth), '-', label='center line v-velocity Re=1000', 
            linewidth=1.2, color='orange')
    ax1.plot(ghia_x_v, ghia_v_re100, '^', markersize=6, markerfacecolor='none',
            color='blue', alpha=0.7, label='Ghia et al. (1982) Re=100')
    ax1.plot(ghia_x_v, ghia_v_re1000, 's', markersize=6, markerfacecolor='none',
            color='red', alpha=0.7, label='Ghia et al. (1982) Re=1000')
    ax1.plot(ghia_x_v, v_sim_1000, 'o', markersize=6, markerfacecolor='none',
            color='black', linewidth=1.5, label='Simulation Re=1000')
    ax1.plot(ghia_x_v, v_sim_100, 'X', markersize=7, markerfacecolor='none',
            color='black', linewidth=1.5, label='Simulation Re=100')
    ax1.set_xlabel('x / X',fontstyle = 'italic', fontsize=11)
    ax1.set_ylabel('v / U',fontstyle = 'italic', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # U-velocity plot
    ax2.plot(y_smooth, cs_u_re100(y_smooth) , '-', label='center line u-velocity Re=100',
            linewidth=1.2, color='lightblue')
    ax2.plot(y_smooth, cs_u_re1000(y_smooth),  '-', label='center line u-velocity Re=1000',
            linewidth=1.2, color='orange')
    ax2.plot(ghia_y_u, ghia_u_re100,  '^', markersize=6, markerfacecolor='none',
            color='blue', alpha=0.7, label='Ghia et al. (1982) Re=100')
    ax2.plot(ghia_y_u, ghia_u_re1000,  's', markersize=6, markerfacecolor='none',
            color='red', alpha=0.7, label='Ghia et al. (1982) Re=1000')
    ax2.plot(ghia_y_u, u_sim_1000, 'o', markersize=6, markerfacecolor='none',
            color='black', linewidth=1.5, label='Simulation Re=1000')
    ax2.plot(ghia_y_u, u_sim_100, 'X', markersize=7, markerfacecolor='none',
            color='black', linewidth=1.5, label='Simulation Re=100')
    ax2.set_ylabel('u / U', fontstyle = 'italic', fontsize=12)
    ax2.set_xlabel('y / Y', fontstyle = 'italic', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

    #The velocity vector field plot
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/grid_1000, Ly/grid_1000
    target_times = [1.4, 3.4, 5.4, 7.4, 9.4, 12.6]  # Target times

    # Create grid for plotting
    x_centers = np.linspace(dx/2, Lx-dx/2, grid_1000)
    y_centers = np.linspace(dy/2, Ly-dy/2, grid_1000)
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    skip = 4
    for idx_plot, t_target in enumerate(target_times):
        if t_target in velocity_snapshots:
            ax = axes[idx_plot]
            data = velocity_snapshots[t_target]
            u_c = data['u_center']
            v_c = data['v_center']
            
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
                    u_c[::skip, ::skip], v_c[::skip, ::skip],
                    color='black', scale=10, width=0.002, headwidth=4, headlength=6)
            ax.set_facecolor('lavender')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f't = {t_target}',fontsize=20)
            ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()
