import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Joint offsets as a Torch tensor [J, 3] in XYZ (mm)
OFFSETS = torch.tensor([
    [0.0, 0.0, 0.0],           # 0  Hips
    [-132.948591, 0.0, 0.0],   # 1  RightUpLeg
    [0.0, -442.894612, 0.0],   # 2  RightLeg
    [0.0, -454.206447, 0.0],   # 3  RightFoot
    [0.0, 0.0, 162.767078],    # 4  RightToeBase
    [0.0, 0.0, 74.999437],     # 5  Site (toe end)

    [132.948826, 0.0, 0.0],    # 6  LeftUpLeg
    [0.0, -442.894413, 0.0],   # 7  LeftLeg
    [0.0, -454.20659, 0.0],    # 8  LeftFoot
    [0.0, 0.0, 162.767426],    # 9  LeftToeBase
    [0.0, 0.0, 74.999948],     # 10 Site (toe end)

    [0.0, 0.1, 0.0],            # 11 Spine
    [0.0, 233.383263, 0.0],     # 12 Spine1
    [0.0, 257.077681, 0.0],     # 13 Neck
    [0.0, 121.134938, 0.0],     # 14 Head
    [0.0, 115.002227, 0.0],     # 15 Site (head end)

    [0.0, 257.077681, 0.0],     # 16 LeftShoulder
    [0.0, 151.034226, 0.0],     # 17 LeftArm
    [0.0, 278.882773, 0.0],     # 18 LeftForeArm
    [0.0, 251.733451, 0.0],     # 19 LeftHand
    [0.0, 0.0, 0.0],            # 20 LeftHandThumb
    [0.0, 0.0, 99.999627],      # 21 Site
    [0.0, 100.000188, 0.0],     # 22 L_Wrist_End
    [0.0, 0.0, 0.0],            # 23 Site

    [0.0, 257.077681, 0.0],     # 24 RightShoulder
    [0.0, 151.031437, 0.0],     # 25 RightArm
    [0.0, 278.892924, 0.0],     # 26 RightForeArm
    [0.0, 251.72868, 0.0],      # 27 RightHand
    [0.0, 0.0, 0.0],            # 28 RightHandThumb
    [0.0, 0.0, 99.999888],      # 29 Site
    [0.0, 137.499922, 0.0],     # 30 R_Wrist_End
    [0.0, 0.0, 0.0],            # 31 Site
], dtype=torch.float32, device=DEVICE)


# Parent indices for each joint, -1 indicates no parent
PARENTS = [
    -1,     # 0  Hips
     0,     # 1  RightUpLeg
     1,     # 2  RightLeg
     2,     # 3  RightFoot
     3,     # 4  RightToeBase
     4,     # 5  Site (toe end)

     0,     # 6  LeftUpLeg
     6,     # 7  LeftLeg
     7,     # 8  LeftFoot
     8,     # 9  LeftToeBase
     9,     # 10 Site (toe end)

     0,     # 11 Spine
    11,     # 12 Spine1
    12,     # 13 Neck
    13,     # 14 Head
    14,     # 15 Site (head end)

    13,     # 16 LeftShoulder
    16,     # 17 LeftArm
    17,     # 18 LeftForeArm
    18,     # 19 LeftHand
    19,     # 20 LeftHandThumb
    20,     # 21 Site
    19,     # 22 L_Wrist_End
    22,     # 23 Site

    13,     # 24 RightShoulder
    24,     # 25 RightArm
    25,     # 26 RightForeArm
    26,     # 27 RightHand
    27,     # 28 RightHandThumb
    28,     # 29 Site
    27,     # 30 R_Wrist_End
    30      # 31 Site
]


# Train/test split of subjects 
SUBJECTS = {
    "train": ["S1", "S5", "S6", "S7", "S8"],
    "test": ["S9", "S11"]
}