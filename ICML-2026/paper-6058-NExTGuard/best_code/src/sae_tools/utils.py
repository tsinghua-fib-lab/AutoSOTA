import os
from datetime import datetime

class FilenameConstructor:
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./results"
    ):
        """
        Initialize the filename constructor.
        """
        self.output_dir = output_dir
        self.safe_model_name = model_path.strip('/').split('/')[-1]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    def file_name(
        self,
        prefix: str,
        dataset_name: str,
        path: str = "predictions",
        extension: str = "json"
    ):
        
        save_dir = os.path.join(self.output_dir, f"{prefix}_{self.safe_model_name}_{self.timestamp}")
        save_dir = os.path.join(save_dir, path)
        os.makedirs(save_dir, exist_ok=True)

        safe_dataset_name = dataset_name.strip('/').split('/')[-1]
        output_file = f"{safe_dataset_name}.{extension}"
        output_file = os.path.join(save_dir, output_file)
        print(f"📁 Saved to file: {output_file}")
        return output_file