"""
diagnose_h5.py
──────────────
Print shapes, axis ranges, and T extremes from the FD HDF5 file.
Run this BEFORE the parity plot to understand the data layout.
"""

import numpy as np
import h5py

H5_PATH = "Results_datadriven/output_path_long.h5"

with h5py.File(H5_PATH, "r") as f:
    print("Top-level keys:", list(f.keys()))
    
    # Grid
    X = np.array(f["structuredGrid/X"])
    Y = np.array(f["structuredGrid/Y"])
    Z = np.array(f["structuredGrid/Z"])
    print(f"\nGrid arrays:")
    print(f"  X: shape={X.shape}, min={X.min():.6f}, max={X.max():.6f}")
    print(f"  Y: shape={Y.shape}, min={Y.min():.6f}, max={Y.max():.6f}")
    print(f"  Z: shape={Z.shape}, min={Z.min():.6f}, max={Z.max():.6f}")
    
    # Check if Z[0] is bottom or top
    print(f"\n  Z[0]  = {Z[0]:.6f}   (bottom or top?)")
    print(f"  Z[-1] = {Z[-1]:.6f}   (bottom or top?)")
    
    # Check path1 keys
    if "path1" in f:
        path1_keys = sorted(f["path1"].keys())
        print(f"\npath1 has {len(path1_keys)} time groups")
        print(f"  First 3: {path1_keys[:3]}")
        
        # Pick one timestep to inspect
        ts = path1_keys[len(path1_keys)//2]  # middle timestep
        print(f"\n  Inspecting timestep: {ts}")
        
        grp = f[f"path1/{ts}"]
        print(f"  Datasets in this group: {list(grp.keys())}")
        
        if "Temperature_xy" in grp:
            T_xy = np.array(grp["Temperature_xy"])
            print(f"\n  Temperature_xy:")
            print(f"    raw shape  = {T_xy.shape}")
            print(f"    squeezed   = {T_xy.squeeze().shape}")
            print(f"    T min/max  = {T_xy.min():.1f} / {T_xy.max():.1f} K")
            
            T_sq = T_xy.squeeze()
            # Check which axis is x (should be 201) and which is y (should be 51)
            print(f"    → axis 0 has {T_sq.shape[0]} points")
            print(f"    → axis 1 has {T_sq.shape[1]} points")
            print(f"    Expected: nx=201, ny=51")
            
            # Where is the hot spot?
            idx = np.unravel_index(T_sq.argmax(), T_sq.shape)
            print(f"    Hot spot at index {idx}")
            
        if "Temperature_xz" in grp:
            T_xz = np.array(grp["Temperature_xz"])
            print(f"\n  Temperature_xz:")
            print(f"    raw shape  = {T_xz.shape}")
            print(f"    squeezed   = {T_xz.squeeze().shape}")
            print(f"    T min/max  = {T_xz.min():.1f} / {T_xz.max():.1f} K")
            
            T_sq = T_xz.squeeze()
            print(f"    → axis 0 has {T_sq.shape[0]} points")
            print(f"    → axis 1 has {T_sq.shape[1]} points")
            print(f"    Expected: nx=201, nz=51")
            
            idx = np.unravel_index(T_sq.argmax(), T_sq.shape)
            print(f"    Hot spot at index {idx}")
            
            # Check: is the hot region near Z[0] or Z[-1]?
            print(f"    T at [:,0]  max = {T_sq[:,0].max():.1f} K  (z=Z[0]={Z[0]:.4f})")
            print(f"    T at [:,-1] max = {T_sq[:,-1].max():.1f} K  (z=Z[-1]={Z[-1]:.4f})")
    
    print("\nDone.")