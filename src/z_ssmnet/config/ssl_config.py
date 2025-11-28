class SSL_Config:    
    def __init__(self):
        self.images_path = "/workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr"
        self.zonal_mask_path = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post"
        self.output_path = "/workdir/SSL/data"
        self.splits_path = "/workdir/nnUNet_raw_data/Task2302_z-nnmnet/splits.json"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
