import os
import random
import shutil
from pathlib import Path

RANDOM_SEED = 42 
VALIDATION_SPLIT = 0.1

current_directory_path = Path.cwd()
data_path = current_directory_path / "data"

train_path = data_path / "train"
test_path = data_path / "test"

valid_path = data_path / "valid"
if not os.path.exists("valid_path"):
    # Create dir
    os.mkdir(valid_path)

    # Read train dataset
    classes = os.listdir(train_path)
    for cls in classes:
        train_class_path = train_path / cls
        samples = os.listdir(train_class_path)
        
        random.seed(RANDOM_SEED)
        random.shuffle(samples)

        id = int(len(samples) * VALIDATION_SPLIT)
        valid_samples = samples[:id]

        valid_class_path = valid_path / cls
        os.mkdir(valid_class_path)
        
        for sample in valid_samples:
            # Copy image from train folder
            shutil.copyfile(
                src=train_class_path / sample, 
                dst=valid_class_path / sample
            )
            # Remove it from train folder
            os.remove(train_class_path / sample)