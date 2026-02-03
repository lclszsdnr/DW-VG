# DW-VG
## 🛠 Installation
Our main code is based on [MDETR](https://github.com/ashkamath/mdetr), please follow their instructions to set up the environment.
## The application of Maveric for coference resolution.
The code we use to obtain pseudo labels with [Maverick](https://github.com/SapienzaNLP/maverick-coref) is located in the Coreference resolution folder. 
## 💪 Multi-Stage Training Configuration
Running training at different stages requires modifications in three places:

1.Model Reference (models/)
Update the cofer_deter module to import and use the stage-specific model implementation from the corresponding file in the transformers_xxx directory.

2.Parameter Training States (main)
In the main.py script, configure the training status of model parameters for each stage (e.g., freezing or unfreezing specific modules).

3.Weight Initialization (main)
Adjust the model weight initialization in the main.py script to load the appropriate pretrained checkpoints for each training stage.


