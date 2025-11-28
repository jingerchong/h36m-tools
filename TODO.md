h36m_kit/
│
├── __init__.py
│
├── skeleton/               # dataset-specific skeleton info & helpers
│   ├── __init__.py
│   ├── h36m_skeleton.json  # joint hierarchy
│   ├── joint_names.json
│   ├── parents.json
│   ├── offsets.json
│   └── (optionally) helper functions to load/access skeletons
│
├── formats.py              # rotation representation conversions
│   ├── quat_to_rot6d()
│   ├── rot6d_to_quat()
│   ├── quat_to_expmap()
│   └── expmap_to_quat()
│
├── fk.py                   # forward kinematics
│   ├── from_quat()         # quaternion sequence → 3D joint positions
│   └── (optional helpers)
│
├── normalization.py        # normalization / unnormalization
│   ├── compute_stats()
│   ├── normalize()
│   └── unnormalize()
│
├── constant_dims.py        # detect/remove/add constant dims
│   ├── find_constant_dims()
│   ├── remove_constant_dims()
│   └── add_constant_dims()
│
├── sliding_window.py       # generate observation/prediction windows
│   ├── create_windows(sequence, obs_len, pred_len)
│
├── dataset.py              # PyTorch Dataset wrapper
│   ├── H36MDataset         # returns (obs_window, pred_window, metadata)
│
├── metadata.py             # dataset metadata
│   ├── fps, subjects, actions, splits, etc.
│
├── file_io.py              # save/load tensors & JSON
│   ├── save_pt(), load_pt()
│   ├── save_json(), load_json()
│
├── visualization/          # minimal visualization for sanity checks
│   ├── __init__.py
│   ├── plot.py             # single-frame skeleton plotting
│   └── animate.py          # sequence animation
│
└── utils.py                # general helper functions



compare expmap
print data