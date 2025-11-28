class ZNNMNet_Config:    
    def __init__(self):
        self.data_path = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post/"
        self.save_path = "/workdir/nnUNet_preprocessed/Task2302_z-nnmnet/nnUNetData_plans_v2.1_stage0/"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
