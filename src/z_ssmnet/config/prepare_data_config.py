
class Prepare_Data_Config:    
    def __init__(self):
        self.taskname = "task-z-nnmnet"
        self.workdir = "workdir"
        self.inputdir = "input"
        self.imagesdir = "images"
        self.labelsdir = "labels"
        self.splits = "picai_pub"

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")