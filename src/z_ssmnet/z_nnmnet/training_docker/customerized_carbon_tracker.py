from carbontracker.tracker import CarbonTracker

class CustomizedCarbonTracker:
    def __init__(self, logdir, enabled=True):
        if enabled:
            self.tracker = CarbonTracker(
                epochs=1,
                ignore_errors=True,
                devices_by_pid=False,
                log_dir=str(logdir),
                verbose=2
            )
        else:
            self.tracker = None

    def __enter__(self):
        if self.enabled:
            self.tracker.epoch_start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.tracker.epoch_end()
            self.tracker.stop()

    @property
    def enabled(self):
        return self.tracker is not None