class Zonal_Segmentation_Config:    
    def __init__(self):
        self.images_path = "/workdir/nnUNet_raw_data/Task2302_z-nnmnet/imagesTr"
        self.images_zonal_path = "/workdir/nnUNet_raw_data/Task2302_z-nnmnet/images_zonal"
        self.zonal_mask_dir = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions/"
        self.zonal_mask_post_dir = "/workdir/results/nnUNet/3d_fullres/Task990_prostate_zonal_Seg/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/predictions_post"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
