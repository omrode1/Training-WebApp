# YOLO Model Training Application

A comprehensive GUI application for training YOLO (You Only Look Once) object detection models with an intuitive, user-friendly interface.


## landing page

![alt text](/images/landing_page.png)

## Features

- **User-friendly Interface**: Step-by-step wizard for configuring and running model training
- **Dataset Management**: Validate and prepare datasets for training
- **Model Configuration**: Select YOLO model variants and configure hyperparameters
- **Training Visualization**: Monitor training progress in real-time
- **Augmentation Control**: Fine-tune data augmentation for better model generalization
- **WandB Integration**: Track experiments with Weights & Biases
- **File Browser**: GUI-based browsing to select dataset folders and model files
- **Training Methods**: Multiple methods to handle threading issues with YOLO training

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd yolo-trainer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Dataset Preparation

The application expects datasets in the following structure:

```
dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

Labels should be in YOLO format:
```
class_id x_center y_center width height
```

## Usage

1. **Dataset Configuration**:
   - Provide the path to your dataset (GUI file browser available)
   - Specify the number of classes and their names
   - Configure validation split

2. **Model Settings**:
   - Choose a YOLO model variant (nano, small, medium, large, xlarge)
   - Set image size for training
   - Select device (CPU/GPU)

3. **Training Parameters**:
   - Configure epochs, batch size, and optimizer
   - Set learning rate and early stopping parameters
   - Enable WandB integration (optional)

4. **Augmentation**:
   - Configure data augmentation parameters for training
   - Adjust geometric transforms, flips, and advanced augmentations

5. **Training**:
   - Review configuration summary
   - Select training method (Command Line is recommended)
   - Start training
   - Monitor progress and results

## Troubleshooting

### "Signal only works in main thread" Error
If you encounter a "signal only works in main thread" error when starting training, select the "Command Line (recommended)" training method. This uses a subprocess approach that avoids the threading issues that can occur when running YOLO training from Streamlit's multi-threaded environment.

## Requirements

- Python 3.8+
- Streamlit
- Ultralytics YOLO v8
- PyTorch
- CUDA-compatible GPU (recommended for faster training)

## License

MIT License

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io)
- [Weights & Biases](https://wandb.ai)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 