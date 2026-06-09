"""
 INSTALL:  pip install cvxpy numpy matplotlib
   (Clarabel ships with cvxpy >= 1.3; ECOS/SCS are alternative free solvers.)
=============================================================================
"""
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import cvxpy as cp
import matplotlib.pyplot as plt

# ===========================================================================
#  SECTION 1: Network Base Values and Line Parameters
# ===========================================================================
Vnom_kV   = 20.0                       # nominal voltage [kV]
Sbase_MVA = 10.0                       # base power [MVA]
Vbase_kV  = Vnom_kV
Ibase_A   = (Sbase_MVA * 1e6) / (np.sqrt(3) * Vbase_kV * 1e3)   # [A]
Zbase_Ohm = (Vbase_kV * 1e3) ** 2 / (Sbase_MVA * 1e6)          # [Ohm]

# 20 kV XLPE cable (default 150 mm2 NA2XS(FL)2Y 12/20 kV)
#   70 mm2 : r=0.443, x=0.121, Imax=190 A
#   150 mm2: r=0.206, x=0.117, Imax=260 A   <-- default
#   240 mm2: r=0.125, x=0.113, Imax=370 A
r_unit  = 0.206                        # [Ohm/km]
x_unit  = 0.117                        # [Ohm/km]
I_max_A = 260.0                        # [A]
line_length_km = np.array([2.5, 3.0, 2.0, 1.5, 2.0])

r_pu = (r_unit * line_length_km) / Zbase_Ohm     # [p.u.]
x_pu = (x_unit * line_length_km) / Zbase_Ohm     # [p.u.]
I_max_pu = I_max_A / Ibase_A
l_max    = I_max_pu ** 2                          # 

nb, nl = 6, 5
branch = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [4, 6]])
fr = branch[:, 0] - 1                  
to = branch[:, 1] - 1                  

# ===========================================================================
#  SECTION 2: Load and PV Profiles definitions.
# ===========================================================================
Td, N = 1.0, 24
t_axis = np.arange(N)

P_load_base = np.array([1.20, 0.80, 1.50, 0.60, 0.90])    # buses 2..6 [MW]
load_profile = np.array([0.45, 0.42, 0.40, 0.38, 0.40, 0.45,
                         0.55, 0.70, 0.80, 0.82, 0.80, 0.78,
                         0.75, 0.72, 0.70, 0.72, 0.78, 0.90,
                         1.00, 0.95, 0.88, 0.78, 0.65, 0.52])
pf_load = 0.92
tan_phi = np.tan(np.arccos(pf_load))

P_load = np.zeros((nb, N))
P_load[1:, :] = np.outer(P_load_base, load_profile) / Sbase_MVA
Q_load = P_load * tan_phi

PV_bus = 3
pv_profile = np.array([0, 0, 0, 0, 0, 0, 0.02, 0.15, 0.40, 0.65, 0.80, 0.88,
                       0.90, 0.85, 0.75, 0.60, 0.35, 0.10, 0, 0, 0, 0, 0, 0])
PV = np.zeros((nb, N))
PV[PV_bus-1, :] = 1.0 * pv_profile / Sbase_MVA # (0-indexed bus)

# ===========================================================================
#  SECTION 3: BESS and Operational Limit Parameters
# ===========================================================================
BESS_bus = 4                           
E_max_pu_max = 10.0 / Sbase_MVA        
DoD_max = 0.85
eta_ch, eta_dch = 0.95, 0.95
C_rate = 0.5                           
V_min_pu, V_max_pu = 0.95, 1.05
v_min, v_max = V_min_pu ** 2, V_max_pu ** 2
bess_b = BESS_bus - 1                   # 0-indexed BESS bus

# ===========================================================================
#  SECTION 4: Optimization Variable Definition
# ===========================================================================
P = cp.Variable((nl, N), name="P")     # active branch flow [p.u.]
Q = cp.Variable((nl, N), name="Q")     # reactive branch flow [p.u.]
v = cp.Variable((nb, N), name="v")     # squared bus voltage [p.u.^2]
l = cp.Variable((nl, N), name="l")     # squared branch current [p.u.^2]

Emax = cp.Variable(nonneg=True, name="Emax")   # BESS energy capacity [p.u.]
Pch  = cp.Variable(N, nonneg=True, name="Pch") # charging power [p.u.]
Pdch = cp.Variable(N, nonneg=True, name="Pdch")# discharging power [p.u.]
Eb   = cp.Variable(N, name="Eb")               # state of charge [p.u.]

# ===========================================================================
#  SECTION 5: Network and BESS Constraints
# ===========================================================================
cons = []


cons += [v[0, :] == 1.0]
cons += [Emax <= E_max_pu_max]


for ij in range(nl):
    cons += [v[to[ij], :] == v[fr[ij], :]
             - 2 * (r_pu[ij] * P[ij, :] + x_pu[ij] * Q[ij, :])
             + (r_pu[ij] ** 2 + x_pu[ij] ** 2) * l[ij, :]]

# --- SOCP cone:  P^2 + Q^2 <= v_i * l   (rotated SOC as a standard SOC) ---
#     ||[2P, 2Q, v_i - l]||_2 <= v_i + l   <=>   P^2 + Q^2 <= v_i * l
for ij in range(nl):
    vi = v[fr[ij], :]
    soc_stack = cp.vstack([2 * P[ij, :], 2 * Q[ij, :], vi - l[ij, :]])  # 3 x N
    cons += [cp.SOC(vi + l[ij, :], soc_stack, axis=0)]                  # N cones


for b in range(1, nb):                 # buses 1..5 (0-indexed) = buses 2..6
    in_br  = np.where(to == b)[0]      
    out_br = np.where(fr == b)[0]      

    Pin  = cp.sum([P[k, :] - r_pu[k] * l[k, :] for k in in_br]) if len(in_br) else 0
    Qin  = cp.sum([Q[k, :] - x_pu[k] * l[k, :] for k in in_br]) if len(in_br) else 0
    Pout = cp.sum([P[k, :] for k in out_br]) if len(out_br) else 0
    Qout = cp.sum([Q[k, :] for k in out_br]) if len(out_br) else 0

    
    Pbess = (Pdch - Pch) if b == bess_b else 0

    cons += [Pin - Pout + Pbess == P_load[b, :] - PV[b, :]]
    cons += [Qin - Qout         == Q_load[b, :]]


cons += [v[1:, :] >= v_min, v[1:, :] <= v_max]
cons += [l >= 0, l <= l_max]
cons += [P[0, :] >= 0]                


cons += [Pch  <= C_rate * Emax]
cons += [Pdch <= C_rate * Emax]
cons += [Eb >= (1 - DoD_max) * Emax, Eb <= Emax]
for t in range(N):
    tn = (t + 1) % N
    cons += [Eb[tn] == Eb[t] + eta_ch * Td * Pch[t] - (Td / eta_dch) * Pdch[t]]

# ===========================================================================
#  SECTION 6: Objective Function and Optimization Solver
# ===========================================================================

loss = Td * cp.sum(cp.multiply(r_pu[:, None], l))
objective = cp.Minimize(loss + 1e-4 * Emax)

prob = cp.Problem(objective, cons)

# Pick a free solver that is installed (Clarabel ships with CVXPY >= 1.3)
free_solvers = [s for s in ("CLARABEL", "ECOS", "SCS") if s in cp.installed_solvers()]
solver = free_solvers[0] if free_solvers else None
print(f"Model 2 (CVXPY): solving SOCP DistFlow with solver = {solver} ...")
prob.solve(solver=solver, verbose=True)

print(f"\nStatus: {prob.status}.  Objective (total losses): {prob.value:.6f} [p.u. h]")

# ===========================================================================
#  SECTION 7: Post-Processing of Optimization Results
# ===========================================================================
P_branch = P.value
Q_branch = Q.value
l_branch = l.value
V_bus    = np.sqrt(np.maximum(v.value, 0.0))
E_BESS   = Eb.value
P_ch     = Pch.value
P_dch    = Pdch.value
P0       = P_branch[0, :]              # substation active power = line 1-2 flow

E_max_opt_MWh = Emax.value * Sbase_MVA
P_loss_pu = r_pu[:, None] * l_branch
P_loss_total_MWh = P_loss_pu.sum() * Sbase_MVA * Td

# ===========================================================================
#  SECTION 8: Numerical Results and SOCP Tightness Check
# ===========================================================================
print("\n========== MODEL 2 (CVXPY) RESULTS ==========")
print(f"Optimal BESS at bus {BESS_bus}: {E_max_opt_MWh:.2f} MWh, "
      f"P_max {C_rate * E_max_opt_MWh:.2f} MW")
print(f"Total EXACT losses: {P_loss_total_MWh:.4f} MWh/day")
print(f"Average substation power: {P0.mean() * Sbase_MVA:.4f} MW")
print("\nBus voltage range [p.u.]:")
for j in range(nb):
    print(f"  Bus {j+1}: {V_bus[j, :].min():.4f} .. {V_bus[j, :].max():.4f}")

print("\nSOCP gap check (should be ~0 -> relaxation tight):")
for t in range(0, N, 8):
    for ij in range(nl):
        vi = 1.0 if fr[ij] == 0 else V_bus[fr[ij], t] ** 2
        gap = P_branch[ij, t] ** 2 + Q_branch[ij, t] ** 2 - vi * l_branch[ij, t]
        print(f"  t={t:2d} line {branch[ij,0]}-{branch[ij,1]}: gap={gap:+.2e}")

# ===========================================================================
#  SECTION 9: Result Visualization and Plots
# ===================================================Result Visualization and Plots========================
fig1, ax = plt.subplots(figsize=(9, 5))
for j in range(nb):
    ax.step(t_axis, V_bus[j, :], where="post", label=f"Bus {j+1}")
ax.axhline(V_min_pu, ls="--", color="r"); ax.axhline(V_max_pu, ls="--", color="r")
ax.set_xlabel("Time [h]"); ax.set_ylabel("V [p.u.]")
ax.set_title("Bus voltages - SOCP DistFlow (CVXPY)")
ax.set_ylim(V_min_pu - 0.02, V_max_pu + 0.02); ax.grid(True); ax.legend(loc="lower right")

fig2, axs = plt.subplots(2, 1, figsize=(9, 6))
axs[0].step(t_axis, E_BESS * Sbase_MVA, where="post")
axs[0].axhline(E_max_opt_MWh, ls="--", color="b")
axs[0].axhline((1 - DoD_max) * E_max_opt_MWh, ls="--", color="r")
axs[0].set_ylabel("E_BESS [MWh]"); axs[0].grid(True)
axs[0].set_title(f"BESS SoC at bus {BESS_bus} (E_max = {E_max_opt_MWh:.2f} MWh)")
axs[1].step(t_axis, (P_dch - P_ch) * Sbase_MVA, where="post")
axs[1].axhline(0, ls=":", color="k")
axs[1].set_xlabel("Time [h]"); axs[1].set_ylabel("P_BESS,net [MW]")
axs[1].set_title("Net BESS injection (+ = discharge)"); axs[1].grid(True)

fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.step(t_axis, P_loss_pu.sum(axis=0) * Sbase_MVA, where="post")
ax3.set_xlabel("Time [h]"); ax3.set_ylabel("P_loss [MW]")
ax3.set_title(f"Exact network losses (total {P_loss_total_MWh:.3f} MWh/day)")
ax3.grid(True)

plt.tight_layout()
plt.show()
