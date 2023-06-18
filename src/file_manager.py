from pathlib import Path

class FileManager:

    
    def __init__(self):
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / 'data'
        self.train_dir = self.data_dir / 'raw'