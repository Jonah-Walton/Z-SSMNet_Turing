# This code is adapted from https://github.com/MrGiovanni/ModelsGenesis/blob/master/infinite_generator_3D.py. 
# The original code is licensed under the attached LICENSE (https://github.com/yuanyuan29/Z-SSMNet/blob/master/src/z_ssmnet/ssl/LICENSE).

class setup_config():
    adc_max = 3000.0
    adc_min = 0.0
    
    def __init__(self, 
            input_rows=None, 
            input_cols=None,
            input_deps=None,
            crop_rows=None, 
            crop_cols=None,
            len_border=None,
            len_border_z=None,
            scale=None,
            DATA_DIR=None,
            train_fold=[0,1,2,3,4],
            valid_fold=[5],
            ):
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.input_deps = input_deps
        self.crop_rows = crop_rows
        self.crop_cols = crop_cols
        self.len_border = len_border
        self.len_border_z = len_border_z
        self.scale = scale
        self.DATA_DIR = DATA_DIR
        self.train_fold = train_fold
        self.valid_fold = valid_fold

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")