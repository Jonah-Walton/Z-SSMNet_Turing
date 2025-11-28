import os
class Training_Config:    
    def __init__(self):
        self.workdir = "/workdir"
        self.imagesdir = os.environ.get('SM_CHANNEL_IMAGES', "/input/images")
        self.labelsdir = os.environ.get('SM_CHANNEL_LABELS', "/input/picai_labels")
        self.outputdir = os.environ.get('SM_MODEL_DIR', "/output")
        # Cross-validation splits. Can be a path to a json file or one of the predefined splits:
        #       picai_pub, picai_pubpriv, picai_pub_nnunet, picai_pubpriv_nnunet, picai_debug
        self.splits = "picai_pub"
        self.nnUNet_tf = 8
        self.nnUNet_tl = 8
        # Directory containing preprocessed input data
        self.preprocesseddir = os.environ.get("SM_CHANNEL_PREPROCESSED", "./preprocessed")
        # Directory where checkpoints will be stored data
        self.preprocesseddir = "./checkpoints"
        self.pretrainedweightsdir = os.environ.get('PRETRAINED_WEIGHTS_DIR', "./input/pretrained")
        self.nnuNet_n_proc_DA = None
        # Folds to train. Default: 0 1 2 3 4
        self.folds = (0, 1, 2, 3, 4)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
