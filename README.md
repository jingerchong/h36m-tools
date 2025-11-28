## H3.6M Preprocessing and Visualization Kit

This repo contains helpful tools to preprocess and understand H3.6M data. Notably, it includes scripts to convert raw D3_Angles downloaded from the official H3.6M website into a variety of SO(3) representations such as quaternions, expmap/axis-angle, Euler angles, rotation matrices, etc. This hopefully mkaes it easier for people to recreate the zip file of expmap transformeed data originally from Jain et al or whatever paper

It also includes scripts to transform from the base/reference representation of quaternions to positions through Forward Kinematics, and a tool to visualize this on a 3D animation plot as a check. In the process, we also filter out and save any constant dims, and compute global statistics such as mean and std for nomrlaization later. and we have saved several config files with our best descriptions of what these are. to save headache for those just getting started parsing h3.6m.

This implementation relies on simple common python tools like pytorch, matplotlib, etc.

## Intall Dependencies

Please go to command line and run the ff

to make suire you chave all the packages installed needed

## Download H3.6M Dataset

Then, navigate to the official H3.6M webpage http://vision.imar.ro/human3.6m/ and create a login to request your own license/acceess to the dataset. you can also use other github repos that download all these data at once can be integrated into this repo in the future.

Once you have access, please login and go to the Download menu tab, downoading 'D3_Angles' for all actions for all subjects. The easiest way I have found to do this is to just go to Training Data, By subject, and for each subject click Poses then D3_Angles. You can then unzip each downloaded tgz into data/raw

The folder structure expected by this script is as follows

data/ raw
- S1
  - MyPoseFeatures
    - D3_Angles
      - Directions 1.cdf
      - Directions.cdf
      - Discussion 1.cdf
      - ...
- S5
  - MyPoseFeatures
    - D3_Angles
      - ...

As a side note, you can look at these files to see what these numbers actually represent. After a bit of digging, I realized that these angels are Cardan/Euler angles in the order of ___ for each angle (labeled in the kseleotn.json). These angels are defined relative to the previous parent angle. Often, the root position (first 3 values) as well as root rotation (next 3 values after) is discarded/ignored when training and evaluating (thru metrics) in motion prediction 


## Preprocessing


## Visualization


## Other Scripts