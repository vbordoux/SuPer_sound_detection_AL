Work in progress repo to use active learning approach to train model based on Surfperch

github: https://github.com/BenUCL/surfperch/tree/surfperch
DOI: https://doi.org/10.48550/arXiv.2404.16436

# Installation
- Clone this repository in your local environment
- Create an environment with Python version >= 3.10
- Activate the environment if needed
- Install dependencies for Perch model:
 pip install git+https://github.com/google-research/perch.git@e27e95344c84601759d4d881289a1c2b53697fa6
- Reinstall tensorflow
 pip uninstall tensorflow
 pip install tensorflow==2.15.0 #(you might want to install a version supporting GPU and compatible with your CUDA and cuDNN versions instead)
 restart kernel if needed (if import tensorflow returns an error)
- Open "SurfPerch_active_learning.ipynb" & select the kernel of your environment
- Provide your data to get started

# Getting started
It might be easier to start with the colab notebook in the orginal repository with more detailled instructions and data provided:
https://colab.research.google.com/github/BenUCL/surfperch/blob/surfperch/SurfPerch_Demo_with_Calling_in_Our_Corals.ipynb#scrollTo=c-iKkbImw_I_

# Working on personal data
Minimal running instructions are in the jupyter notebook "SurfPerch_active_learning.ipynb"

- The notebook is made to use the architecture of folder provided in the github repository and is easier if reproduced

├── README.md
├── SurfPerch_active_learning.ipynb
├── [location_name Data]
│   ├── [deployment_name]
│   │   ├── raw_audio
│   │       └── all_wav_files_here
│   │   └── ref_sound
│   │       ├── [sound1_name]
│   │       │   └── sound1.wav
│   │       └── [sound2_name]
│   │           └── sound2.wav
│   └── SurfPerch-model
│       ...
└── [location_name Outputs]


- Once wav files are in the raw_audio folder and at least one target sound is provided in ref_sound, the notebook can be use to train a model on this sound


