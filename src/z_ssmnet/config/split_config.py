class Split_Config:    
    def __init__(self):
        # Output path to store cross-validation splits, type String
        self.output = ""

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
