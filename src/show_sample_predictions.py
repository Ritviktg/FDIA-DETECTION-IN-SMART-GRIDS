import numpy as np
import json
import sys
import pandapower.networks as pn
import pandapower as pp

# Reconfigure stdout to support unicode icons on Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

def show_predictions_for_grid(grid_name='14', count=20):
    shap_values   = np.load(f'outputs/shap_values_{grid_name}.npy')
    explained_idx = np.load(f'outputs/shap_explained_idx_{grid_name}.npy')
    gt_test       = np.load(f'dataset/gt_test_{grid_name}.npy')

    with open(f'dataset/at_test_{grid_name}.json') as f:
        at_test = json.load(f)

    # Load grid network topology
    if grid_name == '14':
        net = pn.case14()
    else:
        net = pn.case118()
        
    pp.runpp(net, verbose=False)

    line_mapping = {}
    for i, (_, line) in enumerate(net.line.iterrows()):
        try:
            from_idx = net.bus.index.get_loc(line.from_bus)
            to_idx   = net.bus.index.get_loc(line.to_bus)
            line_mapping[i] = (from_idx, to_idx)
        except:
            pass
            
    n_buses = len(net.bus)

    print(f"\n======================================================================================")
    print(f"Sample predictions from IEEE {grid_name}-bus test set (Top-1 and Top-3 localization):")
    print(f"======================================================================================")
    print(f"{'Sample':>6} {'Attack Type':>12} {'True Bus':>10} {'Top-1 Pred':>12} {'Top-1 OK?':>10} {'Top-3 Pred':>22} {'Top-3 OK?':>10}")
    print("-" * 86)

    for i in range(min(count, len(shap_values))):
        sv = shap_values[i]
        true_bus = int(gt_test[explained_idx[i]])
        atype = at_test[explained_idx[i]]
        
        # Calculate bus SHAP attribution by summing connected absolute line SHAP flows
        bus_attributions = np.zeros(n_buses)
        for line_idx, (from_bus, to_bus) in line_mapping.items():
            if line_idx < len(sv):
                bus_attributions[from_bus] += abs(sv[line_idx])
                bus_attributions[to_bus]   += abs(sv[line_idx])
                
        sorted_buses = np.argsort(bus_attributions)[::-1]
        top1 = int(sorted_buses[0])
        top3 = [int(x) for x in sorted_buses[:3]]
        
        if true_bus == -1:
            top1_ok = "❌"
            top3_ok = "❌"
        else:
            top1_ok = "✅" if true_bus == top1 else "❌"
            top3_ok = "✅" if true_bus in top3 else "❌"
            
        print(f"{i+1:>6} {atype:>12} {true_bus:>10} {top1:>12} {top1_ok:>10} {str(top3):>22} {top3_ok:>10}")

if __name__ == '__main__':
    grid = sys.argv[1] if len(sys.argv) > 1 else '14'
    show_predictions_for_grid(grid, count=20)
