# %%
import sys
sys.path.append('../core')  # to import from parent folder

import numpy as onp
from potentials import dist_lin_seg_nonjax

def place_rods_with_stats(num_rods, rod_diameter, container_size, max_attempts=10_000):
    q = onp.zeros((num_rods, 5), dtype=onp.float64)

    failures_before_success = onp.zeros(num_rods, dtype=onp.int64)
    attempts_per_rod = onp.zeros(num_rods, dtype=onp.int64)

    placed = 0
    total_attempts = 0
    early_stop = False

    for i in range(num_rods):
        created = False
        attempts = 0

        while not created and attempts < max_attempts:
            x = onp.random.uniform(-1, 1)
            y = onp.random.uniform(-1, 1)
            z = onp.random.uniform(-1, 1)
            phi = onp.random.uniform(0, onp.pi)
            theta = onp.random.uniform(0, 2 * onp.pi)

            p_i = onp.array([x, y, z])
            p_ii = p_i + 1 * onp.array([
                onp.sin(phi) * onp.cos(theta),
                onp.sin(phi) * onp.sin(theta),
                onp.cos(phi)
            ])

            intersect = False

            # Container check for the new rod (do it always; cheap and clear)
            if (onp.linalg.norm(p_i) > container_size or
                onp.linalg.norm(p_ii) > container_size):
                intersect = True
            else:
                # Check against already placed rods
                for j in range(i):
                    x2, y2, z2, phi2, theta2 = q[j]
                    p_j = onp.array([x2, y2, z2])
                    p_jj = p_j + 1 * onp.array([
                        onp.sin(phi2) * onp.cos(theta2),
                        onp.sin(phi2) * onp.sin(theta2),
                        onp.cos(phi2)
                    ])

                    # Inter-rod distance + container check for the existing rod endpoints
                    if (onp.linalg.norm(p_j) > container_size or
                        onp.linalg.norm(p_jj) > container_size):
                        intersect = True
                        break

                    distance = dist_lin_seg_nonjax(p_i, p_ii, p_j, p_jj)
                    if distance < rod_diameter:
                        intersect = True
                        break

            if not intersect:
                q[i] = onp.array([x, y, z, phi, theta])
                created = True
                placed += 1

            attempts += 1
            total_attempts += 1

        attempts_per_rod[i] = attempts
        # failures before success = attempts - 1 (or = max_attempts if we failed out)
        failures_before_success[i] = attempts - 1

        if attempts == max_attempts and not created:
            print(f"[Early stop] Failed to place rod {i} after {max_attempts} attempts.")
            early_stop = True
            break

        if i % 10 == 0 and created:
            print(f"Rod {i} placed successfully (attempts: {attempts})")

    # Trim arrays if we early-stopped
    if early_stop:
        failures_before_success = failures_before_success[:placed]
        attempts_per_rod = attempts_per_rod[:placed]
        q = q[:placed]

    success_rate_per_attempt = placed / max(total_attempts, 1)
    avg_failures_before_success = failures_before_success.mean() if placed > 0 else onp.nan
    avg_attempts_per_rod = attempts_per_rod.mean() if placed > 0 else onp.nan

    stats = {
        "placed": int(placed),
        "requested": int(num_rods),
        "early_stop": early_stop,
        "total_attempts": int(total_attempts),
        "success_rate_per_attempt": float(success_rate_per_attempt),
        "failures_before_success": failures_before_success,  # per placed rod
        "attempts_per_rod": attempts_per_rod,               # per placed rod
        "avg_failures_before_success": float(avg_failures_before_success),
        "avg_attempts_per_rod": float(avg_attempts_per_rod),
    }

    print(
        f"[Summary] placed={placed}/{num_rods}, "
        f"total_attempts={total_attempts}, "
        f"success_rate/attempt={success_rate_per_attempt:.4f}, "
        f"avg_failures_before_success={avg_failures_before_success:.2f}, "
        f"avg_attempts/rod={avg_attempts_per_rod:.2f}, "
        f"early_stop={early_stop}"
    )

    return q, stats

if __name__ == "__main__":
    
    C = 1.
    seg_len = 1.
    alpha = 100
    rod_diameter = seg_len / alpha
    # n_jam / 
    n_jam = (2*C)**3 / (onp.pi * (rod_diameter/2) * seg_len**2)
    import numpy as np


    nn = np.geomspace(1, n_jam*1.5, 10).astype(int)
    yy = []
    for i in range(len(nn)):

        q, stats = place_rods_with_stats(
        num_rods=nn[i], rod_diameter=0.01, container_size=C, max_attempts=20000
        )

        yy.append(stats["success_rate_per_attempt"])

    from matplotlib import pyplot as plt
    plt.plot(nn, yy, 'o-')
    plt.show()

    rho = 1
    # save
    output_dict = {
        "n_jam": n_jam,
        "nn": nn,
        "yy": yy,
        "rho": rho,
        "alpha": alpha,
        "stats": stats,
    }

    onp.savez(f"failure_ratio_alpha{alpha}_rho{rho}.npz", **output_dict)

    # Example quick looks:
    # print("Placed:", stats["placed"])
    # print("Avg failures before success:", stats["avg_failures_before_success"])
    # print("Success rate per attempt:", stats["success_rate_per_attempt"])
# %%
    xx = nn / n_jam
    from matplotlib import pyplot as plt
    plt.figure(figsize=(4,3))
    plt.plot(xx, yy, 'o-')
    plt.xlabel(r'$N/N_{jam}$', fontsize=14)
    plt.ylabel(r'$1/p_{success}$', fontsize=14)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.grid(True, which="both", ls="--", lw=0.5)
# %%
    import numpy as np
    pth = '/Users/yeonsu/GitHub/entanglement-optimization/analysis/failure_ratio_alpha100_rho1.npz'
    dd = np.load(pth)
    nn = dd['nn']
    yy = dd['yy']
    n_jam = dd['n_jam']
    rho = dd['rho']
    alpha = dd['alpha']
    

    from matplotlib import pyplot as plt
    plt.figure(figsize=(4,3))
    plt.plot(nn/n_jam, 1/yy, 'o-')

# %%
    