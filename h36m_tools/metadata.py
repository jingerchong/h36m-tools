import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Number of rotational joints in raw H3.6M D3_Angles
# CDF has 78 dims = 26 joints × 3 angles per joint
# The first 3 dims correspond to root translation and are not actual rotations,
# so we ignore them when loading, leaving 25 joints with 3 rotation angles each
NUM_JOINTS = 25


# Joint offsets as a Torch tensor [J, 3] in XYZ (mm)
OFFSETS = torch.tensor([
    [0.0, 0.0, 0.0],           # 0  Hips
    [-132.948591, 0.0, 0.0],   # 1  RightUpLeg
    [0.0, -442.894612, 0.0],   # 2  RightLeg
    [0.0, -454.206447, 0.0],   # 3  RightFoot
    [0.0, 0.0, 162.767078],    # 4  RightToeBase

    [132.948826, 0.0, 0.0],    # 5  LeftUpLeg
    [0.0, -442.894413, 0.0],   # 6  LeftLeg
    [0.0, -454.20659, 0.0],    # 7  LeftFoot
    [0.0, 0.0, 162.767426],    # 8  LeftToeBase

    [0.0, 0.1, 0.0],            # 9 Spine
    [0.0, 233.383263, 0.0],     # 10 Spine1
    [0.0, 257.077681, 0.0],     # 11 Neck
    [0.0, 121.134938, 0.0],     # 12 Head

    [0.0, 257.077681, 0.0],     # 13 LeftShoulder
    [0.0, 151.034226, 0.0],     # 14 LeftArm
    [0.0, 278.882773, 0.0],     # 15 LeftForeArm
    [0.0, 251.733451, 0.0],     # 16 LeftHand
    [0.0, 0.0, 0.0],            # 17 LeftHandThumb
    [0.0, 100.000188, 0.0],     # 18 L_Wrist_End

    [0.0, 257.077681, 0.0],     # 19 RightShoulder
    [0.0, 151.031437, 0.0],     # 20 RightArm
    [0.0, 278.892924, 0.0],     # 21 RightForeArm
    [0.0, 251.72868, 0.0],      # 22 RightHand
    [0.0, 0.0, 0.0],            # 23 RightHandThumb
    [0.0, 137.499922, 0.0],     # 24 R_Wrist_End
], dtype=torch.float32, device=DEVICE)


# Parent indices for each joint, -1 indicates no parent
PARENTS = [
    -1,     # 0  Hips
     0,     # 1  RightUpLeg
     1,     # 2  RightLeg
     2,     # 3  RightFoot
     3,     # 4  RightToeBase

     0,     # 5  LeftUpLeg
     5,     # 6  LeftLeg
     6,     # 7  LeftFoot
     7,     # 8  LeftToeBase

     0,     # 9 Spine
     9,     # 10 Spine1
    10,     # 11 Neck
    11,     # 12 Head

    11,     # 13 LeftShoulder
    13,     # 14 LeftArm
    14,     # 15 LeftForeArm
    15,     # 16 LeftHand
    16,     # 17 LeftHandThumb
    16,     # 18 L_Wrist_End

    11,     # 19 RightShoulder
    19,     # 20 RightArm
    20,     # 21 RightForeArm
    21,     # 22 RightHand
    22,     # 23 RightHandThumb
    22,     # 24 R_Wrist_End
]


# Names of joints split by kinematic chain
JOINT_NAMES = [
    "Hips",            # 0
    "RightUpLeg",      # 1
    "RightLeg",        # 2
    "RightFoot",       # 3
    "RightToeBase",    # 4

    "LeftUpLeg",       # 5
    "LeftLeg",         # 6
    "LeftFoot",        # 7
    "LeftToeBase",     # 8

    "Spine",           # 9
    "Spine1",          # 10
    "Neck",            # 11
    "Head",            # 12

    "LeftShoulder",    # 13
    "LeftArm",         # 14
    "LeftForeArm",     # 15
    "LeftHand",        # 16
    "LeftHandThumb",   # 17
    "L_Wrist_End",     # 18

    "RightShoulder",   # 19
    "RightArm",        # 20
    "RightForeArm",    # 21
    "RightHand",       # 22
    "RightHandThumb",  # 23
    "R_Wrist_End"      # 24
]


# Pairs of joints indices (right, left)
RIGHT_LEFT_JOINTS_IDX = [
    (1, 5),    # RightUpLeg <-> LeftUpLeg
    (2, 6),    # RightLeg <-> LeftLeg
    (3, 7),    # RightFoot <-> LeftFoot
    (4, 8),    # RightToeBase <-> LeftToeBase
    (19, 13),  # RightShoulder <-> LeftShoulder
    (20, 14),  # RightArm <-> LeftArm
    (21, 15),  # RightForeArm <-> LeftForeArm
    (22, 16),  # RightHand <-> LeftHand
    (23, 17),  # RightHandThumb <-> LeftHandThumb
    (24, 18),  # R_Wrist_End <-> L_Wrist_End
]


# Train/test split of subjects 
SUBJECTS = {
    "train": ["S1", "S5", "S6", "S7", "S8"],
    "test": ["S9", "S11"]
}


# expmap site add in