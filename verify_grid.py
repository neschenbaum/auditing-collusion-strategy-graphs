
import numpy as np
from env import CalvanoAdjPriceGridEnv

def test_grid():
    env = CalvanoAdjPriceGridEnv()
    grid = env.price_grid
    
    print(f"Grid size: {len(grid)}")
    print(f"Grid: {grid}")
    
    # Check for Nash and Monopoly
    nash = 1.47
    monopoly = 1.92
    
    has_nash = np.any(np.isclose(grid, nash, atol=1e-9))
    has_monopoly = np.any(np.isclose(grid, monopoly, atol=1e-9))
    
    print(f"\nTarget Nash: {nash}")
    print(f"Has Nash? {has_nash}")
    
    print(f"Target Monopoly: {monopoly}")
    print(f"Has Monopoly? {has_monopoly}")
    
    if has_nash and has_monopoly:
        print("\nSUCCESS: Both prices present.")
    else:
        print("\nFAILURE: Missing target prices.")

if __name__ == "__main__":
    test_grid()
