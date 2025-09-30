Code used for the experiments of the manuscript "Rapid Fish Sound Detection Using Human-in-The-Loop Active Learning" Bordoux et al. 2025
Preprint available here: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5403106

The code is mainly based on the development done by Perch team:
https://github.com/google-research/perch
Agile Modeling, the method used in this repository is published in Dumoulin et al. (2025), here:  https://arxiv.org/abs/2505.03071
Please cite appropriately upon reuse.

# Installation
- Clone this repository on your computer
- Create a virtual environment with Python version >= 3.10
- Activate the environment if needed
- Install dependencies for this project by running (require poetry to be installed):

```bash
python -m pip install --upgrade pip
poetry install
```
The code was developed and tested on Linux computers only.

# Running agile modeling on personal data
- The tree view bellow explains how the project folder can be organised to facilitate reuse of the scripts
- Place audio files in dataset_name Data/deployment_name/raw_audio
- Place the target sound in dataset_name Data/deployment_name/ref_sound/target_sound_name/ (one folder per sound type)
- Create config_dict.json based on config_dict_example.json with the correct path, name, and model
- Run create_embeddings.ipynb (only once, GPU recommended for Perch and Surfperch)
- Run agile_modeling_training.ipynb

# Folder structure
.
├── poetry.lock  
├── pyproject.toml  
├── utils_agile_model.py  
├── .gitignore  
├── README.md  
├── dataset_name Data  
│   └── deployment_name  
│       ├── annotations  
│       ├── raw_audio  
│       ├── ref_sound  
│       │   ├── target_sound_name  
│       └── test_set  
├── dataset_name Outputs  
├── agile_modeling_training.ipynb  
├── config_dict_example.json  
├── create_embeddings.ipynb  
├── create_test_set_data.ipynb  
├── experiments_script.ipynb  
├── figures_results_experiment.ipynb  
├── models  
|   ├── BirdNET-model  
|   |   ├── BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite  
|   ├── Perch-model  
|   |   ├── assets  
|   |   ├── info_model  
|   |   ├── saved_model.pb  
|   |   ├── variables  
|   ├── SurfPerch-model  
|   |   ├── assets  
|   |   ├── info_model  
|   |   ├── saved_model.pb  
|   |   ├── variables  


