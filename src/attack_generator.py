import pandapower as pp
import pandapower.networks as pn
import numpy as np
import os
import json

def load_grid(grid_name):
    if grid_name == '14':
        net = pn.case14()
    elif grid_name == '118':
        net = pn.case118()
    pp.runpp(net, verbose=False)
    return net

def build_H_dc(net):
    n_bus  = len(net.bus)
    n_line = len(net.line)
    scale = 100.0 if n_bus == 14 else 1000.0
    H = np.zeros((n_line, n_bus))
    for i, (_, line) in enumerate(net.line.iterrows()):
        try:
            from_idx = net.bus.index.get_loc(line.from_bus)
            to_idx   = net.bus.index.get_loc(line.to_bus)
            x_ohm = line.x_ohm_per_km * line.length_km
            x_ohm = max(x_ohm, 0.01)
            b = scale / x_ohm
            H[i, from_idx] = +b
            H[i, to_idx]   = -b
        except Exception as e:
            print(f"Line {i} skipped: {e}")
    return H

def inject_fdia_valid(z, H, attack_type, 
                      target_bus=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_buses = H.shape[1]
    if target_bus is None:
        target_bus = np.random.randint(0, n_buses)
    
    c = np.zeros(n_buses)
    
    if attack_type == 'scaling':
        c[target_bus] = np.random.uniform(0.1, 0.3)
    elif attack_type == 'random':
        c[target_bus] = np.random.normal(0, 0.2)
    elif attack_type == 'pulse':
        c[target_bus] = np.random.choice([-0.3, 0.3])
    elif attack_type == 'ramping':
        c[target_bus] = np.random.uniform(0.05, 0.25)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    a = H @ c
    z_attacked = z + a
    return z_attacked, target_bus, c

def verify_stealth(z, z_attacked, H, threshold=1e-4):
    H_pinv     = np.linalg.pinv(H)
    r_clean    = z - H @ H_pinv @ z
    r_attacked = z_attacked - H @ H_pinv @ z_attacked
    diff = abs(np.linalg.norm(r_clean) 
               - np.linalg.norm(r_attacked))
    return diff < threshold

def generate_valid_attacks(X_normal, H, n_per_type=5000,
                           random_state=42):
    np.random.seed(random_state)
    attack_types = ['pulse', 'random', 'scaling', 'ramping']
    
    attacked_samples  = []
    ground_truth_buses = []
    attack_type_labels = []
    stealth_pass = 0
    stealth_total = 0
    
    for attack_type in attack_types:
        count = 0
        attempts = 0
        
        while count < n_per_type and attempts < n_per_type * 5:
            attempts += 1
            idx = np.random.randint(0, len(X_normal))
            z   = X_normal[idx].copy()
            
            z_size   = len(z)
            H_n_rows = H.shape[0]
            use_size = min(z_size, H_n_rows)
            z_use    = z[:use_size]
            
            try:
                z_att, bus, c = inject_fdia_valid(
                    z_use, H, attack_type
                )
                
                z_full = z.copy()
                z_full[:use_size] = z_att
                
                if (not np.any(np.isnan(z_full)) and 
                    not np.any(np.isinf(z_full))):
                    
                    attacked_samples.append(z_full)
                    ground_truth_buses.append(bus)
                    attack_type_labels.append(attack_type)
                    
                    stealth_total += 1
                    if verify_stealth(z_use, z_att, H):
                        stealth_pass += 1
                    
                    count += 1
                    
            except Exception as e:
                continue
        
        print(f"  {attack_type}: {count} valid samples generated")
    
    print(f"\nStealth verification: "
          f"{stealth_pass}/{stealth_total} passed")
    
    return (np.array(attacked_samples), 
            np.array(ground_truth_buses),
            attack_type_labels)

if __name__ == '__main__':
    X_normal_14  = np.load('dataset/X_normal_14.npy')
    X_normal_118 = np.load('dataset/X_normal_118.npy')

    net14  = load_grid('14')
    net118 = load_grid('118')
    H14    = build_H_dc(net14)
    H118   = build_H_dc(net118)

    print(f"H14  shape: {H14.shape}")
    print(f"H118 shape: {H118.shape}")

    print("\nGenerating valid attacks for IEEE 14-bus...")
    X_attacked_14, gt_buses_14, types_14 = generate_valid_attacks(
        X_normal_14, H14, n_per_type=5000
    )

    print("\nGenerating valid attacks for IEEE 118-bus...")
    X_attacked_118, gt_buses_118, types_118 = generate_valid_attacks(
        X_normal_118, H118, n_per_type=5000
    )

    os.makedirs('dataset', exist_ok=True)

    np.save('dataset/X_attacked_14.npy',       X_attacked_14)
    np.save('dataset/X_attacked_118.npy',      X_attacked_118)
    np.save('dataset/gt_buses_14.npy',         gt_buses_14)
    np.save('dataset/gt_buses_118.npy',        gt_buses_118)

    with open('dataset/attack_types_14.json', 'w') as f:
        json.dump(types_14, f)
    with open('dataset/attack_types_118.json', 'w') as f:
        json.dump(types_118, f)

    print(f"\nX_attacked_14  shape: {X_attacked_14.shape}")
    print(f"X_attacked_118 shape: {X_attacked_118.shape}")
    print(f"gt_buses_14    shape: {gt_buses_14.shape}")
    print(f"gt_buses_118   shape: {gt_buses_118.shape}")
    print(f"Attack types 14  — unique: " f"{list(set(types_14))}")
    print(f"Attack types 118 — unique: " f"{list(set(types_118))}")
