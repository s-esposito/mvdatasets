import yaml
from pathlib import Path


def save_yaml(
    data_dict: dict,
    file_path: Path
):
    """
    Save function. Convert Paths to strings.
    
    Args:
        data: Data to save.
        file_path: Path to save the data.
    """
    # Convert Path objects to strings
    serializable_data = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in data_dict.items()
    }
    # Save to YAML file
    with open(file_path, "w") as f:
        yaml.dump(serializable_data, f)


def load_yaml(file_path: Path) -> dict:
    """
    Load function. Convert strings whose key ends with '_path' back to Paths.

    Args:
        file_path: Path to load the data.
    Returns:
        Loaded deserialized_data data as a dictionary.
    """
    with open(file_path, "r") as f:
        loaded_data = yaml.safe_load(f)
    # Convert strings back to Path objects
    data_dict = {
        key: Path(value) if key.endswith("_path") else value
        for key, value in loaded_data.items()
    }
    return data_dict