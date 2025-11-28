class SSL_Generator_Config:    
    def __init__(self):
        # fold of subset (0-5), int
        self.fold = ""
        self.input_rows = 64
        self.input_cols = 64
        self.input_deps = 16
        self.crop_rows = 64
        self.crop_cols = 64
        # directory containing subset0..subset5 folders, str
        self.data = " "
        # output directory for generated cubes, str
        self.save = ""
        self.scale = 12

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
