import os
import yaml
import wandb
import random
import shutil
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import subprocess
from typing import Dict, Optional, List
import argparse
from tqdm import tqdm
import sys



class YOLOTrainer:
    def __init__(self):
        print("Initializing YOLO Trainer...")
        self.default_models = {
            "n": "yolov8n.pt",
            "s": "yolov8s.pt",
            "m": "yolov8m.pt",
            "l": "yolov8l.pt",
            "x": "yolov8x.pt"
        }
        
        self.config = {
            "model": "",
            "epochs": 100,
            "batch_size": 16,
            "imgsz": 640,
            "project_name": "YOLO_Training",
            "data_yaml_path": "",
            "train_dir": "",
            "val_dir": "",
            "val_split": 0.2,
            "patience": 50,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "device": "",
            "cache": False,
            "workers": 8,
            "resume": False,
            "exist_ok": False,
            "freeze": 0,


            #hyperparameters
            "degrees": 0.2,
            "translate": 0.01,
            "scale": 0.25,
            "shear": 0.2,
            "perspective": 0.001,
            "flipud": 0,
            "fliplr": 0.35,
            "mosaic": 0.02,
            "mixup": 0.2, 
            "sync" : False, 
            "verbose" : False

        }
        print("Initialization complete.")

    def validate_dataset_structure(self, dataset_path: str) -> bool:
        """Validate the dataset directory structure"""
        dataset_path = Path(dataset_path)
        print(f"\nValidating dataset structure at: {dataset_path}")
        
        # Check if main directory exists
        if not dataset_path.exists():
            print(f"Error: Dataset directory {dataset_path} does not exist!")
            return False
            
        # Check for images and labels directories
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not images_dir.exists():
            print(f"Error: Images directory not found at {images_dir}")
            return False
        if not labels_dir.exists():
            print(f"Error: Labels directory not found at {labels_dir}")
            return False
            
        # Check if there are any images
        image_files = list(images_dir.glob('*.*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if not image_files:
            print("Error: No image files found in images directory!")
            return False
            
        print(f"Found {len(image_files)} images in dataset.")
        return True

    def setup_wandb(self) -> None:
        """Initialize WandB project"""
        print("\nInitializing WandB...")
        try:
            wandb.init(
                project=self.config["project_name"],
                config=self.config,
                resume="allow",
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print("WandB initialization successful!")
        except Exception as e:
            print(f"Warning: WandB initialization failed: {str(e)}")
            print("Training will continue without WandB logging")

    def split_dataset(self, dataset_path: str) -> None:
        """Split dataset into train and validation sets"""
        print("\nSplitting dataset...")
        try:
            dataset_path = Path(dataset_path)
            
            # Create train and val directories
            train_dir = dataset_path.parent / "train"
            val_dir = dataset_path.parent / "val"
            
            for split_dir in [train_dir, val_dir]:
                for subdir in ['images', 'labels']:
                    (split_dir / subdir).mkdir(parents=True, exist_ok=True)
                    print(f"Created directory: {split_dir / subdir}")
            
            # Get all image files
            image_files = list((dataset_path / 'images').glob('*.*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            
            if not image_files:
                raise Exception("No image files found in dataset!")
            
            # Split files
            train_files, val_files = train_test_split(
                image_files, 
                test_size=self.config['val_split'],
                random_state=42
            )
            
            print(f"Training images: {len(train_files)}")
            print(f"Validation images: {len(val_files)}")
            
            # Copy files to respective directories
            for files, dest_dir in [(train_files, train_dir), (val_files, val_dir)]:
                for img_path in tqdm(files, desc=f"Copying to {dest_dir.name}"):
                    # Copy image
                    shutil.copy2(img_path, dest_dir / 'images' / img_path.name)
                    
                    # Copy corresponding label
                    label_path = dataset_path / 'labels' / f"{img_path.stem}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, dest_dir / 'labels' / f"{img_path.stem}.txt")
            
            self.config['train_dir'] = str(train_dir)
            self.config['val_dir'] = str(val_dir)
            
        except Exception as e:
            print(f"Error during dataset splitting: {str(e)}")
            raise

    def create_data_yaml(self, dataset_path: str, num_classes: int, class_names: List[str]) -> str:
        """Create data.yaml file for training"""
        print("\nCreating data.yaml file...")
        try:
            data_yaml = {
                'train': str(Path(self.config['train_dir']) / 'images'),
                'val': str(Path(self.config['val_dir']) / 'images'),
                'nc': num_classes,
                'names': class_names
            }
            
            yaml_path = Path(dataset_path).parent / 'data.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, sort_keys=False)
            
            print(f"Created data.yaml at: {yaml_path}")
            print("YAML contents:")
            print(yaml.dump(data_yaml))
            
            self.config['data_yaml_path'] = str(yaml_path)
            return str(yaml_path)
            
        except Exception as e:
            print(f"Error creating data.yaml: {str(e)}")
            raise

    def get_user_config(self) -> None:
        """Get training configuration from user"""
        print("\nYOLO Training Configuration")
        print("==========================")
        
        try:
            # Model selection
            print("\nAvailable models:")
            for key, model in self.default_models.items():
                print(f"{key}: {model}")
            model_key = input("\nSelect model (n/s/m/l/x/custom) [default: n]: ").lower() or 'n'
            if model_key == 'custom':
                self.config['model'] = input("Enter full path to custom model weights for transfer learning: ")
                if not Path(self.config['model']).is_file():
                    print(f"Error: Custom model file {self.config['model']} does not exist!")
                    sys.exit(1)
            else:
                self.config['model'] = self.default_models.get(model_key, self.default_models['n'])
            self.config['model'] = self.default_models.get(model_key, self.default_models['n'])
            
            # Training parameters
            self.config['epochs'] = int(input(f"Number of epochs [default: {self.config['epochs']}]: ") 
                                      or self.config['epochs'])
            self.config['batch_size'] = int(input(f"Batch size [default: {self.config['batch_size']}]: ") 
                                          or self.config['batch_size'])
            self.config['imgsz'] = int(input(f"Image size [default: {self.config['imgsz']}]: ") 
                                      or self.config['imgsz'])
            self.config['val_split'] = float(input(f"Validation split [default: {self.config['val_split']}]: ") 
                                           or self.config['val_split'])
            
            # Advanced parameters
            print("\nAdvanced parameters (press Enter to use defaults):")
            self.config['patience'] = int(input(f"Early stopping patience [default: {self.config['patience']}]: ") 
                                        or self.config['patience'])
            self.config['optimizer'] = input(f"Optimizer (SGD/Adam) [default: {self.config['optimizer']}]: ").upper() or self.config['optimizer']
            self.config['lr0'] = float(input(f"Initial learning rate [default: {self.config['lr0']}]: ") 
                                      or self.config['lr0'])
            self.config['device'] = input("Device (cpu/0/0,1,2,3...) [default: auto]: ") or ""
            self.config['freeze'] = int(input(f"Freeze layers (0=none, 1=backbone, 2=all) [default: {self.config['freeze']}]: ") 
                                      or self.config['freeze'])
            
            # Project name
            self.config['project_name'] = input(f"WandB project name [default: {self.config['project_name']}]: ") or self.config['project_name']
            
            print("\nConfiguration complete.")
            print("Current config:")
            for key, value in self.config.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error during configuration: {str(e)}")
            raise

    def train(self) -> None:
        """Start training process"""
        try:
            print("\nImporting ultralytics...")
            from ultralytics import YOLO
            
            print("\nStarting training...")
            print(f"Using model: {self.config['model']}")
            print(f"Data YAML: {self.config['data_yaml_path']}")
            
            # Initialize model
            model = YOLO(self.config['model'])
            
            # Start training
            results = model.train(
                data=self.config['data_yaml_path'],
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch_size'],
                device=self.config['device'],
                optimizer=self.config['optimizer'],
                patience=self.config['patience'],
                lr0=self.config['lr0'],
                lrf=self.config['lrf'],
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay'],
                warmup_epochs=self.config['warmup_epochs'],
                warmup_momentum=self.config['warmup_momentum'],
                warmup_bias_lr=self.config['warmup_bias_lr'],
                cache=self.config['cache'],
                workers=self.config['workers'],
                resume=self.config['resume'],
                exist_ok=self.config['exist_ok'],
                project=self.config['project_name'], 
                augment=True,
                degrees=self.config['degrees'],
                translate=self.config['translate'],
                scale=self.config['scale'],
                shear=self.config['shear'],
                perspective=self.config['perspective'],
                flipud=self.config['flipud'],
                fliplr=self.config['fliplr'],
                mosaic=self.config['mosaic'],
                mixup=self.config['mixup']
            )
            
            print("\nTraining completed successfully!")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

def main():
    print("Starting YOLO Training Pipeline...")
    
    parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--names', type=str, nargs='+', required=True, help='Class names')
    
    try:
        args = parser.parse_args()
        print(f"\nReceived arguments:")
        print(f"Dataset path: {args.dataset}")
        print(f"Number of classes: {args.classes}")
        print(f"Class names: {args.names}")
        
        trainer = YOLOTrainer()
        
        # Validate dataset structure
        if not trainer.validate_dataset_structure(args.dataset):
            print("Dataset validation failed. Exiting...")
            sys.exit(1)
        
        # Get user configuration
        trainer.get_user_config()
        
        # Split dataset
        trainer.split_dataset(args.dataset)
        
        # Create data.yaml
        trainer.create_data_yaml(args.dataset, args.classes, args.names)
        
        # Initialize WandB
        trainer.setup_wandb()
        
        # Start training
        trainer.train()
        
        # Close WandB run
        wandb.finish()
        
        print("\nTraining pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()