from rich import print
from mvdatasets.utils.printing import (
    print_error,
    print_warning,
    print_info,
    print_log,
    print_success,
)
from mvdatasets.camera import Camera
from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.tensor_reel import TensorReel

# Set the custom exception handler
import sys


def custom_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler to print detailed information for uncaught exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Allow program to exit quietly on Ctrl+C
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Format the exception message
    message = f"{exc_type.__name__}: {exc_value}"
    # Pass detailed exception info to the print_error function
    print_error(message, exc_type, exc_value, exc_traceback)


# Set the custom exception handler globally
sys.excepthook = custom_exception_handler