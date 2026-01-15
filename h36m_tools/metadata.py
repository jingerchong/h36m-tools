import torch

from h36m_tools.files import DATA_DEVICE


# Number of rotational joints in raw H3.6M D3_Angles
# CDF has 78 dims = 26 joints × 3 angles per joint
# The first 3 dims correspond to root translation and are not actual rotations,
# so we ignore them when loading, leaving 25 joints with 3 rotation angles each
NUM_JOINTS = 25


# Joint offsets as a Torch tensor [J, 3] in XYZ (mm)
# Adapted from https://github.com/TUM-AAS/motron-cvpr22/blob/master/config/h36m_skeleton.yaml
OFFSETS = torch.tensor([
    [0.0, 0.0, 0.0],           # 0  Hips
    [-132.948591, 0.0, 0.0],   # 1  RightUpLeg
    [0.0, -442.894612, 0.0],   # 2  RightLeg
    [0.0, -454.206447, 0.0],   # 3  RightFoot
    [0.0, 0.0, 162.767078],    # 4  RightToeBase
    [0.0, 0.0, 74.999437],     # 5  Site
    [132.948826, 0.0, 0.0],    # 6  LeftUpLeg
    [0.0, -442.894413, 0.0],   # 7  LeftLeg
    [0.0, -454.20659, 0.0],    # 8  LeftFoot
    [0.0, 0.0, 162.767426],    # 9  LeftToeBase
    [0.0, 0.0, 74.999948],     # 10 Site
    [0.0, 0.1, 0.0],           # 11 Spine
    [0.0, 233.383263, 0.0],    # 12 Spine1
    [0.0, 257.077681, 0.0],    # 13 Neck
    [0.0, 121.134938, 0.0],    # 14 Head
    [0.0, 115.002227, 0.0],    # 15 Site
    [0.0, 257.077681, 0.0],    # 16 LeftShoulder
    [0.0, 151.034226, 0.0],    # 17 LeftArm
    [0.0, 278.882773, 0.0],    # 18 LeftForeArm
    [0.0, 251.733451, 0.0],    # 19 LeftHand
    [0.0, 0.0, 0.0],           # 20 LeftHandThumb
    [0.0, 0.0, 99.999627],     # 21 Site
    [0.0, 100.000188, 0.0],    # 22 LeftWristEnd
    [0.0, 0.0, 0.0],           # 23 Site
    [0.0, 257.077681, 0.0],    # 24 RightShoulder
    [0.0, 151.031437, 0.0],    # 25 RightArm
    [0.0, 278.892924, 0.0],    # 26 RightForeArm
    [0.0, 251.72868, 0.0],     # 27 RightHand
    [0.0, 0.0, 0.0],           # 28 RightHandThumb
    [0.0, 0.0, 99.999888],     # 29 Site
    [0.0, 137.499922, 0.0],    # 30 RightWristEnd
    [0.0, 0.0, 0.0],           # 31 Site
], dtype=torch.float32, device=DATA_DEVICE)


# Parent indices for each joint
PARENTS = [
    -1,  # 0  Hips
     0,  # 1  RightUpLeg
     1,  # 2  RightLeg
     2,  # 3  RightFoot
     3,  # 4  RightToeBase
     4,  # 5  Site
     0,  # 6  LeftUpLeg
     6,  # 7  LeftLeg
     7,  # 8  LeftFoot
     8,  # 9  LeftToeBase
     9,  # 10 Site
     0,  # 11 Spine
    11,  # 12 Spine1
    12,  # 13 Neck
    13,  # 14 Head
    14,  # 15 Site
    12,  # 16 LeftShoulder
    16,  # 17 LeftArm
    17,  # 18 LeftForeArm
    18,  # 19 LeftHand
    19,  # 20 LeftHandThumb
    20,  # 21 Site
    19,  # 22 LeftWristEnd
    22,  # 23 Site
    12,  # 24 RightShoulder
    24,  # 25 RightArm
    25,  # 26 RightForeArm
    26,  # 27 RightHand
    27,  # 28 RightHandThumb
    28,  # 29 Site
    27,  # 30 RightWristEnd
    30   # 31 Site
]


JOINT_NAMES = [
    "Hips",            # 0
    "RightUpLeg",      # 1
    "RightLeg",        # 2
    "RightFoot",       # 3
    "RightToeBase",    # 4
    "Site_RFoot",      # 5
    "LeftUpLeg",       # 6
    "LeftLeg",         # 7
    "LeftFoot",        # 8
    "LeftToeBase",     # 9
    "Site_LFoot",      # 10
    "Spine",           # 11
    "Spine1",          # 12
    "Neck",            # 13
    "Head",            # 14
    "Site_Head",       # 15
    "LeftShoulder",    # 16
    "LeftArm",         # 17
    "LeftForeArm",     # 18
    "LeftHand",        # 19
    "LeftHandThumb",   # 20
    "Site_LHand",      # 21
    "LeftWristEnd",    # 22
    "Site_LWrist",     # 23
    "RightShoulder",   # 24
    "RightArm",        # 25
    "RightForeArm",    # 26
    "RightHand",       # 27
    "RightHandThumb",  # 28
    "Site_RHand",      # 29
    "RightWristEnd",   # 30
    "Site_RWrist",     # 31
]


RIGHT_LEFT_JOINTS_IDX = [
    (1, 6),    # RightUpLeg - LeftUpLeg
    (2, 7),    # RightLeg - LeftLeg
    (3, 8),    # RightFoot - LeftFoot
    (4, 9),    # RightToeBase - LeftToeBase
    (5, 10),   # Site_RFoot - Site_LFoot
    (24, 16),  # RightShoulder - LeftShoulder
    (25, 17),  # RightArm - LeftArm
    (26, 18),  # RightForeArm - LeftForeArm
    (27, 19),  # RightHand - LeftHand
    (28, 20),  # RightHandThumb - LeftHandThumb
    (29, 21),  # Site_RHand - Site_LHand
    (30, 22),  # RightWristEnd - LeftWristEnd
    (31, 23),  # Site_RWrist - Site_LWrist
]


# Other formats such as raw H3.6M D3_Positions and expmap zip
# have extra placeholder "Site" joints
SITE_JOINTS = [5, 10, 15, 21, 23, 29, 31]
TOTAL_JOINTS = NUM_JOINTS + len(SITE_JOINTS)


# Indices based on joint count in D3_Angles (SITE_JOINTS excluded)
STATIC_JOINTS = [0, 17, 18, 23, 24]


# Train/test split of subjects 
PROTOCOL_1 = {
    "train": ["S1", "S5", "S6", "S7", "S8"],
    "test": ["S9", "S11"]
}
PROTOCOL_2 = {
    "train": ["S1", "S6", "S7", "S8", "S9", "S11"],
    "test": ["S5"]
}

DOWNSAMPLE_FACTOR = 2
RAW_FPS = 50