import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import random
import shutil
from PIL import Image
import sys
import subprocess
import multiprocessing
import tempfile
import time
from datetime import datetime

# Import YOLOTrainer from app.py
from app import YOLOTrainer

# Set page configuration
st.set_page_config(
    page_title="YOLO Trainer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #4CAF50;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #2196F3;
        margin-bottom: 0.5rem;
    }
    .info-text {
        font-size: 1rem;
        color: #555;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .folder-picker {
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if not already present
if 'trainer' not in st.session_state:
    st.session_state.trainer = YOLOTrainer()
if 'dataset_validated' not in st.session_state:
    st.session_state.dataset_validated = False
if 'data_yaml_created' not in st.session_state:
    st.session_state.data_yaml_created = False
if 'training_started' not in st.session_state:
    st.session_state.training_started = False
if 'training_method' not in st.session_state:
    st.session_state.training_method = "Command Line (recommended)"
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'recent_folders' not in st.session_state:
    st.session_state.recent_folders = []
if 'selected_folder' not in st.session_state:
    st.session_state.selected_folder = ""

# Main application header
st.markdown("<div class='main-header'>YOLO Model Training Application</div>", unsafe_allow_html=True)
st.markdown(
    "A comprehensive GUI application for training YOLO (You Only Look Once) object detection models with an intuitive, user-friendly interface."
)

# Sidebar for navigation and app state
with st.sidebar:
    st.markdown("<div class='sub-header'>Navigation</div>", unsafe_allow_html=True)
    
    # Progress indicator
    steps = [
        "1. Dataset Configuration",
        "2. Model Settings",
        "3. Training Parameters",
        "4. Augmentation",
        "5. Training"
    ]
    
    for i, step in enumerate(steps, 1):
        if i < st.session_state.current_step:
            st.success(step)
        elif i == st.session_state.current_step:
            st.info(step)
        else:
            st.markdown(step)
    
    st.divider()
    
    # About section
    st.markdown("<div class='sub-header'>About</div>", unsafe_allow_html=True)
    st.markdown(
        "This application simplifies the process of training YOLO object detection models "
        "by providing an intuitive interface for configuring and monitoring the training process."
    )

# Helper functions for directory browsing
def add_to_recent_folders(folder_path):
    """Add folder to recent folders list and ensure no duplicates"""
    if folder_path and os.path.isdir(folder_path):
        if folder_path in st.session_state.recent_folders:
            st.session_state.recent_folders.remove(folder_path)
        st.session_state.recent_folders.insert(0, folder_path)
        # Keep only the 5 most recent folders
        st.session_state.recent_folders = st.session_state.recent_folders[:5]

def get_parent_directory(path):
    """Get the parent directory of the given path"""
    return os.path.abspath(os.path.join(path, os.pardir))

def explore_directory(base_path):
    """Create a directory explorer UI for the given base path"""
    try:
        # Show current path
        st.text(f"Current directory: {base_path}")
        
        # Option to go up one level
        parent_dir = get_parent_directory(base_path)
        if parent_dir != base_path and os.access(parent_dir, os.R_OK):
            if st.button("📁 ..", key="parent_dir"):
                return parent_dir
        
        # List all subdirectories
        subdirs = [d for d in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, d)) and not d.startswith('.')]
        subdirs.sort()
        
        # Display subdirectories as buttons
        cols = st.columns(3)
        for i, subdir in enumerate(subdirs):
            with cols[i % 3]:
                full_path = os.path.join(base_path, subdir)
                if st.button(f"📁 {subdir}", key=f"dir_{subdir}"):
                    return full_path
        
        # Option to select current directory
        if st.button("✅ Select Current Directory", key="select_current"):
            return base_path
            
        return base_path
    except Exception as e:
        st.error(f"Error accessing directory: {str(e)}")
        return base_path

def explore_model_files(base_path):
    """Create a file explorer UI for selecting model files"""
    try:
        # Show current path
        st.text(f"Current directory: {base_path}")
        
        # Option to go up one level
        parent_dir = get_parent_directory(base_path)
        if parent_dir != base_path and os.access(parent_dir, os.R_OK):
            if st.button("📁 ..", key="parent_dir_model"):
                return parent_dir, None
        
        # List all subdirectories and model files
        items = []
        for item in os.listdir(base_path):
            full_path = os.path.join(base_path, item)
            if os.path.isdir(full_path) and not item.startswith('.'):
                items.append((item, True))  # (name, is_dir)
            elif item.endswith(('.pt', '.pth', '.weights')):
                items.append((item, False))  # (name, is_dir)
        
        items.sort(key=lambda x: (not x[1], x[0]))  # Sort directories first, then files
        
        # Display items as buttons
        cols = st.columns(3)
        for i, (item_name, is_dir) in enumerate(items):
            with cols[i % 3]:
                full_path = os.path.join(base_path, item_name)
                icon = "📁" if is_dir else "🔶"
                if st.button(f"{icon} {item_name}", key=f"model_{item_name}"):
                    if is_dir:
                        return full_path, None
                    else:
                        return base_path, full_path
        
        return base_path, None
    except Exception as e:
        st.error(f"Error accessing directory: {str(e)}")
        return base_path, None

# Main application logic based on current step
def render_step_1():
    """Dataset Configuration Step"""
    st.markdown("<div class='sub-header'>Dataset Configuration</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset path input with directory browser
        st.markdown("### Dataset Path")
        
        # Show text input for manual path entry
        dataset_path = st.text_input(
            "Enter dataset path manually",
            value=st.session_state.selected_folder,
            help="Path to your dataset directory containing images and labels folders"
        )
        
        # Directory browser
        with st.expander("Browse for Dataset Folder", expanded=False):
            st.markdown("<div class='folder-picker'>", unsafe_allow_html=True)
            
            # Show tabs for recent folders and directory browser
            tab1, tab2 = st.tabs(["Recent Folders", "Browse Directories"])
            
            with tab1:
                # Recent folders
                if st.session_state.recent_folders:
                    st.markdown("Select a recently used folder:")
                    for folder in st.session_state.recent_folders:
                        if st.button(f"📁 {folder}", key=f"recent_{folder}"):
                            dataset_path = folder
                            st.session_state.selected_folder = folder
                else:
                    st.info("No recent folders. Browse directories or enter a path manually.")
            
            with tab2:
                # Directory browser
                st.markdown("Navigate to your dataset folder:")
                
                # Start with home directory if no path is selected
                if not hasattr(st.session_state, 'current_browse_path'):
                    st.session_state.current_browse_path = os.path.expanduser("~")
                
                # Display the directory explorer
                new_path = explore_directory(st.session_state.current_browse_path)
                if new_path != st.session_state.current_browse_path:
                    st.session_state.current_browse_path = new_path
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # If a path was selected via browser, update the dataset_path
        if dataset_path and dataset_path != st.session_state.selected_folder:
            st.session_state.selected_folder = dataset_path
            add_to_recent_folders(dataset_path)
        
        # Class information
        st.markdown("### Class Information")
        num_classes = st.number_input("Number of Classes", min_value=1, value=1, step=1)
        
        # Dynamic input for class names
        class_names = []
        for i in range(int(num_classes)):
            class_name = st.text_input(f"Class {i} Name", key=f"class_{i}")
            if class_name:
                class_names.append(class_name)
        
        # Validation split slider
        val_split = st.slider(
            "Validation Split",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Percentage of dataset to use for validation"
        )
        st.session_state.trainer.config['val_split'] = val_split
        
    with col2:
        st.markdown("### Dataset Structure")
        st.markdown(
            """
            Your dataset should follow this structure:
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
            
            Label format should be YOLO format:
            ```
            class_id x_center y_center width height
            ```
            """
        )
        
        # Preview dataset if path is valid
        if dataset_path and os.path.isdir(dataset_path):
            images_dir = os.path.join(dataset_path, "images")
            labels_dir = os.path.join(dataset_path, "labels")
            
            if os.path.isdir(images_dir) and os.path.isdir(labels_dir):
                try:
                    # Count images and labels
                    image_files = [f for f in os.listdir(images_dir) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    label_files = [f for f in os.listdir(labels_dir) 
                                  if f.lower().endswith('.txt')]
                    
                    st.success(f"Found {len(image_files)} images and {len(label_files)} labels")
                    
                    # Show sample image if available
                    if image_files:
                        sample_img_path = os.path.join(images_dir, image_files[0])
                        try:
                            img = Image.open(sample_img_path)
                            st.markdown("#### Sample Image")
                            st.image(img, caption=f"Sample: {image_files[0]}", width=250)
                        except Exception as e:
                            st.warning(f"Could not preview image: {str(e)}")
                except Exception as e:
                    st.error(f"Error reading dataset: {str(e)}")
        
        # Validate button
        if st.button("Validate Dataset", type="primary"):
            if not dataset_path:
                st.error("Please provide dataset path")
            elif not class_names or len(class_names) != num_classes:
                st.error("Please provide all class names")
            else:
                # Validate dataset structure
                with st.spinner("Validating dataset structure..."):
                    is_valid = st.session_state.trainer.validate_dataset_structure(dataset_path)
                    if is_valid:
                        st.session_state.dataset_validated = True
                        st.session_state.dataset_path = dataset_path
                        st.session_state.num_classes = num_classes
                        st.session_state.class_names = class_names
                        st.success("Dataset validation successful!")
                        
                        # Split dataset and create YAML
                        with st.spinner("Splitting dataset into train and validation sets..."):
                            st.session_state.trainer.split_dataset(dataset_path)
                            
                        with st.spinner("Creating data.yaml..."):
                            yaml_path = st.session_state.trainer.create_data_yaml(
                                dataset_path, num_classes, class_names
                            )
                            st.session_state.data_yaml_created = True
                            st.success(f"Created data.yaml at: {yaml_path}")
                    else:
                        st.error("Dataset validation failed. Please check the structure.")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.session_state.dataset_validated and st.session_state.data_yaml_created:
            if st.button("Next: Model Settings"):
                st.session_state.current_step = 2
                st.rerun()

def render_step_2():
    """Model Settings Step"""
    st.markdown("<div class='sub-header'>Model Settings</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Model selection
        model_options = {
            "yolov8n.pt": "YOLOv8 Nano - Fastest, smallest",
            "yolov8s.pt": "YOLOv8 Small - Fast, balanced",
            "yolov8m.pt": "YOLOv8 Medium - Balanced",
            "yolov8l.pt": "YOLOv8 Large - More accurate, slower",
            "yolov8x.pt": "YOLOv8 XLarge - Most accurate, slowest",
            "custom": "Custom model (for transfer learning)"
        }
        
        model_type = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        
        if model_type == "custom":
            # Allow file selection for custom model
            with st.expander("Select Custom Model", expanded=True):
                custom_model_path = st.text_input(
                    "Custom Model Path",
                    help="Path to your custom model weights for transfer learning"
                )
                
                # Add tabs for different ways to select a model
                model_tab1, model_tab2, model_tab3 = st.tabs(["Upload Model", "Browse Files", "Recent Models"])
                
                with model_tab1:
                    uploaded_model = st.file_uploader(
                        "Upload Model Weights (.pt file)",
                        type=["pt", "pth", "weights"],
                        help="Upload a custom YOLOv8 weights file for transfer learning"
                    )
                    
                    if uploaded_model:
                        # Save the uploaded model to a temporary location
                        model_save_path = os.path.join("models", uploaded_model.name)
                        os.makedirs("models", exist_ok=True)
                        
                        with open(model_save_path, "wb") as f:
                            f.write(uploaded_model.getbuffer())
                        
                        custom_model_path = model_save_path
                        st.success(f"Uploaded model saved to: {model_save_path}")
                
                with model_tab2:
                    # Model file browser
                    st.markdown("Navigate to your model file:")
                    
                    # Initialize browse path for models if not present
                    if not hasattr(st.session_state, 'model_browse_path'):
                        st.session_state.model_browse_path = os.path.expanduser("~")
                    if not hasattr(st.session_state, 'recent_models'):
                        st.session_state.recent_models = []
                    
                    # Browse for model files
                    new_path, selected_model = explore_model_files(st.session_state.model_browse_path)
                    
                    if new_path != st.session_state.model_browse_path:
                        st.session_state.model_browse_path = new_path
                        st.rerun()
                    
                    if selected_model:
                        custom_model_path = selected_model
                        st.session_state.selected_model = selected_model
                        
                        # Add to recent models
                        if selected_model in st.session_state.recent_models:
                            st.session_state.recent_models.remove(selected_model)
                        st.session_state.recent_models.insert(0, selected_model)
                        st.session_state.recent_models = st.session_state.recent_models[:5]  # Keep only 5 most recent
                        
                        st.success(f"Selected model: {os.path.basename(selected_model)}")
                
                with model_tab3:
                    # Recent models
                    if hasattr(st.session_state, 'recent_models') and st.session_state.recent_models:
                        st.markdown("Select a recently used model:")
                        for model in st.session_state.recent_models:
                            if os.path.exists(model):
                                if st.button(f"🔶 {os.path.basename(model)}", key=f"recent_model_{model}"):
                                    custom_model_path = model
                                    st.session_state.selected_model = model
                    else:
                        st.info("No recent models. Browse or upload a model file.")
        
            st.session_state.trainer.config['model'] = custom_model_path
        else:
            st.session_state.trainer.config['model'] = model_type
        
        # Image size
        img_size_options = [320, 416, 512, 640, 768, 896, 1024, 1280]
        img_size = st.select_slider(
            "Image Size",
            options=img_size_options,
            value=640,
            help="Input image size for training (higher values require more GPU memory)"
        )
        st.session_state.trainer.config['imgsz'] = img_size
        
        # Device selection
        device_options = ["", "cpu", "0", "0,1", "0,1,2,3"]
        device = st.selectbox(
            "Device",
            options=device_options,
            index=0,
            format_func=lambda x: "Auto (recommended)" if x == "" else f"CUDA {x}" if x.isdigit() or "," in x else x.upper(),
            help="Device to run training on (empty for auto-selection)"
        )
        st.session_state.trainer.config['device'] = device
        
    with col2:
        st.markdown("### Model Information")
        
        if model_type != "custom":
            st.markdown(
                f"""
                **Selected Model**: {model_options[model_type]}
                
                **Image Size**: {img_size}×{img_size}
                
                **Device**: {"Auto" if device == "" else f"CUDA {device}" if device.isdigit() or "," in device else device.upper()}
                
                YOLOv8 models balance speed and accuracy:
                - Nano/Small: Faster inference, less accurate
                - Medium: Balanced performance
                - Large/XLarge: More accurate, slower inference
                """
            )
        else:
            st.markdown(
                f"""
                **Custom Model**: Transfer learning
                
                **Image Size**: {img_size}×{img_size}
                
                **Device**: {"Auto" if device == "" else f"CUDA {device}" if device.isdigit() or "," in device else device.upper()}
                
                Using a custom model allows you to leverage pre-trained weights
                for your specific dataset, potentially improving performance.
                """
            )
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Previous: Dataset Configuration"):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.button("Next: Training Parameters"):
            st.session_state.current_step = 3
            st.rerun()

def render_step_3():
    """Training Parameters Step"""
    st.markdown("<div class='sub-header'>Training Parameters</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Basic training parameters
        st.markdown("### Basic Parameters")
        
        epochs = st.slider(
            "Epochs",
            min_value=1,
            max_value=500,
            value=100,
            help="Number of training epochs"
        )
        st.session_state.trainer.config['epochs'] = epochs
        
        batch_size = st.select_slider(
            "Batch Size",
            options=[1, 2, 4, 8, 16, 32, 64, 128],
            value=16,
            help="Batch size for training (higher values require more GPU memory)"
        )
        st.session_state.trainer.config['batch_size'] = batch_size
        
        # Advanced parameters (collapsible)
        with st.expander("Advanced Parameters"):
            optimizer = st.selectbox(
                "Optimizer",
                options=["SGD", "Adam", "AdamW"],
                index=0,
                help="Optimization algorithm"
            )
            st.session_state.trainer.config['optimizer'] = optimizer
            
            lr0 = st.number_input(
                "Initial Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.01,
                format="%.4f",
                help="Initial learning rate"
            )
            st.session_state.trainer.config['lr0'] = lr0
            
            lrf = st.number_input(
                "Final Learning Rate Factor",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                format="%.4f",
                help="Final learning rate = initial learning rate * factor"
            )
            st.session_state.trainer.config['lrf'] = lrf
            
            patience = st.slider(
                "Early Stopping Patience",
                min_value=0,
                max_value=100,
                value=50,
                help="Epochs to wait for improvement before early stopping (0 to disable)"
            )
            st.session_state.trainer.config['patience'] = patience
            
            freeze = st.slider(
                "Freeze Layers",
                min_value=0,
                max_value=24,
                value=0,
                help="Number of layers to freeze (0: none, 1+: freeze first n layers)"
            )
            st.session_state.trainer.config['freeze'] = freeze
        
        # WandB integration
        with st.expander("WandB Integration"):
            use_wandb = st.checkbox(
                "Enable WandB Logging",
                value=True,
                help="Log training metrics to Weights & Biases"
            )
            
            if use_wandb:
                project_name = st.text_input(
                    "WandB Project Name",
                    value="YOLO_Training",
                    help="Name of the WandB project for logging"
                )
                st.session_state.trainer.config['project_name'] = project_name
    
    with col2:
        st.markdown("### Parameter Recommendations")
        st.markdown(
            f"""
            **Training Duration**: {epochs} epochs
            
            **Resources**: Using batch size {batch_size}
            
            **Recommendations**:
            - For small datasets (<1000 images), try 200-300 epochs
            - For larger datasets, 100 epochs may be sufficient
            - If training plateaus, reduce learning rate or increase patience
            - For transfer learning, consider freezing early layers
            
            **Hardware Requirements**:
            - Selected image size: {st.session_state.trainer.config['imgsz']}×{st.session_state.trainer.config['imgsz']}
            - Selected batch size: {batch_size}
            - Higher batch sizes and image sizes require more GPU memory
            """
        )
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Previous: Model Settings"):
            st.session_state.current_step = 2
            st.rerun()
    with col3:
        if st.button("Next: Augmentation", type="primary"):
            st.session_state.current_step = 4
            st.rerun()

def render_step_4():
    """Augmentation Step"""
    st.markdown("<div class='sub-header'>Data Augmentation</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Augmentation Parameters")
        st.markdown(
            "Data augmentation helps improve model generalization by creating variations of training images."
        )
        
        # Geometric transformations
        st.markdown("#### Geometric Transformations")
        
        degrees = st.slider(
            "Rotation (±degrees)",
            min_value=0.0,
            max_value=45.0,
            value=10.0,
            help="Maximum rotation in degrees"
        )
        st.session_state.trainer.config['degrees'] = degrees / 45.0  # Normalize to 0-1 range
        
        translate = st.slider(
            "Translation",
            min_value=0.0,
            max_value=0.2,
            value=0.1,
            step=0.01,
            help="Maximum translation as a fraction of image size"
        )
        st.session_state.trainer.config['translate'] = translate
        
        scale = st.slider(
            "Scale",
            min_value=0.0,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Maximum scale variation"
        )
        st.session_state.trainer.config['scale'] = scale
        
        shear = st.slider(
            "Shear",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Maximum shear angle in degrees"
        )
        st.session_state.trainer.config['shear'] = shear
        
        perspective = st.slider(
            "Perspective",
            min_value=0.0,
            max_value=0.001,
            value=0.0005,
            step=0.0001,
            format="%.4f",
            help="Perspective distortion"
        )
        st.session_state.trainer.config['perspective'] = perspective
        
        # Flips and mixups
        st.markdown("#### Flips and Advanced Augmentations")
        
        fliplr = st.slider(
            "Horizontal Flip Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability of horizontal flip"
        )
        st.session_state.trainer.config['fliplr'] = fliplr
        
        flipud = st.slider(
            "Vertical Flip Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="Probability of vertical flip (usually 0 for natural images)"
        )
        st.session_state.trainer.config['flipud'] = flipud
        
        mosaic = st.slider(
            "Mosaic Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Probability of using mosaic augmentation (combining 4 images)"
        )
        st.session_state.trainer.config['mosaic'] = mosaic
        
        mixup = st.slider(
            "Mixup Probability",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Probability of using mixup augmentation (blending images)"
        )
        st.session_state.trainer.config['mixup'] = mixup
    
    with col2:
        st.markdown("### Augmentation Preview")
        
        # Add a visual preview of augmentations
        st.markdown("#### Augmentation Effects")
        
        # Display a visual reference for augmentations
        augmentation_effects = {
            "Rotation": "https://pytorch.org/vision/stable/_images/sphx_glr_plot_transforms_002.png",
            "Translation": "https://pytorch.org/vision/stable/_images/sphx_glr_plot_transforms_003.png",
            "Scale": "https://pytorch.org/vision/stable/_images/sphx_glr_plot_transforms_004.png",
            "Horizontal Flip": "https://pytorch.org/vision/stable/_images/sphx_glr_plot_transforms_005.png"
        }
        
        selected_effect = st.selectbox(
            "View augmentation example",
            options=list(augmentation_effects.keys())
        )
        
        if selected_effect in augmentation_effects:
            st.image(
                augmentation_effects[selected_effect],
                caption=f"Example of {selected_effect}",
                width=250
            )
        
        st.markdown("#### Recommended Settings")
        st.markdown(
            """
            - For general object detection: Use defaults
            - For aerial/satellite imagery: Increase rotation, enable vertical flips
            - For medical imaging: Reduce augmentation strength
            - For small datasets: Increase augmentation strength
            """
        )
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Previous: Training Parameters"):
            st.session_state.current_step = 3
            st.rerun()
    with col3:
        if st.button("Next: Training", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

def run_training_in_separate_process(config, data_yaml_path):
    """
    Run YOLO training in a separate process to avoid signal handling issues
    with Streamlit's threading model.
    """
    # Create a temporary YAML file with the training configuration
    config_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    config_path = config_file.name
    
    # Copy all necessary parameters to the config
    training_config = {
        'model': config['model'],
        'data': data_yaml_path,
        'epochs': config['epochs'],
        'imgsz': config['imgsz'],
        'batch': config['batch_size'],
        'device': config['device'],
        'optimizer': config['optimizer'],
        'patience': config['patience'],
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': config['warmup_momentum'],
        'warmup_bias_lr': config['warmup_bias_lr'],
        'project': config['project_name'],
        'name': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'augment': True,
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'mixup': config['mixup']
    }
    
    # Save the config to the temporary file
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f)
    
    # Execute the YOLO training command in a separate process
    from ultralytics import YOLO
    
    def train_process():
        try:
            model = YOLO(config['model'])
            model.train(cfg=config_path)
            return True
        except Exception as e:
            print(f"Training error: {str(e)}")
            return False
    
    # Run the training in a separate process
    process = multiprocessing.Process(target=train_process)
    process.start()
    
    # Wait for completion
    progress_message = st.empty()
    
    while process.is_alive():
        progress_message.info("Training in progress... (This may take a while)")
        # Update progress information here if possible
        time.sleep(5)  # Check status every 5 seconds
    
    process.join()
    
    # Clean up
    try:
        os.unlink(config_path)
    except:
        pass
    
    if process.exitcode == 0:
        return True
    else:
        return False

def run_training_cli(config, data_yaml_path):
    """
    Alternative approach: Run YOLO training via command line subprocess.
    This avoids the signal handling issues completely.
    
    This method solves the "signal only works in main thread" error that occurs
    when trying to run YOLO training directly from Streamlit, which uses a multi-threaded
    environment. By using a subprocess, we isolate the training process and avoid threading issues.
    """
    # Create a temporary YAML file with the training configuration
    config_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
    config_path = config_file.name
    
    # Copy all necessary parameters to the config
    training_config = {
        'model': config['model'],
        'data': data_yaml_path,
        'epochs': config['epochs'],
        'imgsz': config['imgsz'],
        'batch': config['batch_size'],
        'device': config['device'],
        'optimizer': config['optimizer'],
        'patience': config['patience'],
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': config['warmup_momentum'],
        'warmup_bias_lr': config['warmup_bias_lr'],
        'project': config['project_name'],
        'name': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'exist_ok': True,
        'augment': True,
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'mixup': config['mixup']
    }
    
    # Save the config to the temporary file
    with open(config_path, 'w') as f:
        yaml.dump(training_config, f)
    
    # Build the yolo command
    yolo_cmd = f"yolo train model={config['model']} cfg={config_path}"
    
    try:
        # Execute YOLO training command in a subprocess
        process = subprocess.Popen(
            yolo_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Set up placeholders for real-time output
        status_area = st.empty()
        output_area = st.empty()
        
        # Accumulate output
        output_lines = []
        
        # Stream the output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_lines.append(output.strip())
                # Show the last few lines in the output area
                output_area.code('\n'.join(output_lines[-20:]))
                status_area.info(f"Training in progress... (Epoch progress visible below)")
        
        # Get return code
        return_code = process.wait()
        
        # Clean up
        try:
            os.unlink(config_path)
        except:
            pass
        
        if return_code == 0:
            return True, '\n'.join(output_lines)
        else:
            error_output = process.stderr.read()
            return False, error_output
            
    except Exception as e:
        try:
            os.unlink(config_path)
        except:
            pass
        return False, str(e)

def render_step_5():
    """Training Step"""
    st.markdown("<div class='sub-header'>Training</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Training Configuration Summary")
        
        # Dataset info
        st.markdown("#### Dataset")
        st.markdown(
            f"""
            - **Dataset Path**: {getattr(st.session_state, 'dataset_path', 'Not set')}
            - **Classes**: {getattr(st.session_state, 'num_classes', 0)} ({', '.join(getattr(st.session_state, 'class_names', []))})
            - **Validation Split**: {st.session_state.trainer.config['val_split']}
            """
        )
        
        # Model info
        st.markdown("#### Model")
        st.markdown(
            f"""
            - **Model**: {st.session_state.trainer.config['model']}
            - **Image Size**: {st.session_state.trainer.config['imgsz']}×{st.session_state.trainer.config['imgsz']}
            - **Device**: {"Auto" if st.session_state.trainer.config['device'] == "" else st.session_state.trainer.config['device']}
            """
        )
        
        # Training parameters
        st.markdown("#### Training Parameters")
        st.markdown(
            f"""
            - **Epochs**: {st.session_state.trainer.config['epochs']}
            - **Batch Size**: {st.session_state.trainer.config['batch_size']}
            - **Optimizer**: {st.session_state.trainer.config['optimizer']}
            - **Learning Rate**: {st.session_state.trainer.config['lr0']}
            - **Early Stopping Patience**: {st.session_state.trainer.config['patience']}
            """
        )
        
        # Augmentation summary
        st.markdown("#### Augmentation Settings")
        st.markdown(
            f"""
            - **Geometric**: Rotation ±{int(st.session_state.trainer.config['degrees']*45)}°, Translation {st.session_state.trainer.config['translate']}, Scale {st.session_state.trainer.config['scale']}
            - **Flips**: Horizontal {int(st.session_state.trainer.config['fliplr']*100)}%, Vertical {int(st.session_state.trainer.config['flipud']*100)}%
            - **Advanced**: Mosaic {int(st.session_state.trainer.config['mosaic']*100)}%, Mixup {int(st.session_state.trainer.config['mixup']*100)}%
            """
        )
        
        # Add selection for training method
        training_method = st.radio(
            "Training Method",
            options=["Command Line (recommended)", "Subprocess"],
            index=0,
            help="The command line method uses yolo CLI and tends to work better with Streamlit"
        )
        
        # Start training button
        if not st.session_state.training_started:
            if st.button("Start Training", type="primary"):
                st.session_state.training_started = True
                st.session_state.training_method = training_method
                st.rerun()
        else:
            # Display training progress
            st.markdown("### Training Progress")
            
            # Display training progress elements
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            status_text = st.empty()
            output_area = st.empty()
            
            # Start training
            try:
                with st.spinner("Initializing training..."):
                    status_text.text("Setting up training environment...")
                    
                if st.session_state.training_method == "Command Line (recommended)":
                    success, output = run_training_cli(
                        st.session_state.trainer.config,
                        st.session_state.trainer.config['data_yaml_path']
                    )
                else:
                    success = run_training_in_separate_process(
                        st.session_state.trainer.config,
                        st.session_state.trainer.config['data_yaml_path']
                    )
                    output = "Training completed via subprocess"
                
                # Process results
                if success:
                    progress_placeholder.progress(100)
                    status_text.success("Training completed successfully!")
                    
                    # Create folder for results
                    results_path = os.path.join(os.path.dirname(getattr(st.session_state, 'dataset_path', '.')), 
                                              st.session_state.trainer.config['project_name'])
                    
                    # Show results
                    st.markdown("### Training Results")
                    
                    if os.path.exists(results_path):
                        st.success(f"Model results saved to: {results_path}")
                        
                        # Try to find result graphs
                        result_files = []
                        for root, dirs, files in os.walk(results_path):
                            for file in files:
                                if file.endswith(('.png', '.jpg')) and ('results' in file or 'PR_curve' in file or 'confusion_matrix' in file):
                                    result_files.append(os.path.join(root, file))
                        
                        if result_files:
                            st.markdown("#### Performance Graphs")
                            for img_path in result_files[:4]:  # Show up to 4 result images
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, caption=os.path.basename(img_path), use_column_width=True)
                                except:
                                    pass
                        else:
                            st.warning("No result graphs found. Check the results directory manually.")
                    else:
                        st.warning(f"Results directory not found at {results_path}")
                        
                    # Show performance metrics
                    st.markdown(
                        """
                        #### Performance Metrics
                        - Mean Average Precision (mAP)
                        - Precision-Recall curves
                        - Confusion matrix
                        - Inference speed
                        """
                    )
                else:
                    progress_placeholder.empty()
                    status_text.error("Training process encountered an error.")
                    st.error("Training failed. See error details below:")
                    st.code(output)
                    st.session_state.training_started = False
                    
            except Exception as e:
                st.error(f"Error during training setup: {str(e)}")
                st.error("If you see 'signal only works in main thread' error, try the Command Line training method.")
                st.session_state.training_started = False
    
    with col2:
        st.markdown("### Training Information")
        st.markdown(
            """
            **What happens during training:**
            
            1. The model loads the selected pre-trained weights
            2. Dataset is prepared with the specified augmentations
            3. Training proceeds for the set number of epochs
            4. The model is evaluated on the validation set
            5. Early stopping may trigger if performance plateaus
            6. Best weights are saved automatically
            
            **Training time depends on:**
            - Dataset size
            - Model complexity
            - Hardware (CPU/GPU)
            - Image size
            - Batch size
            
            **Monitor training with WandB:**
            If enabled, you can view real-time training metrics and visualizations on the Weights & Biases platform.
            """
        )
        
        # Add restart button when training is done
        if st.session_state.training_started:
            if st.button("Start New Training", key="restart"):
                # Reset relevant session state variables
                st.session_state.training_started = False
                st.session_state.current_step = 1
                st.session_state.dataset_validated = False
                st.session_state.data_yaml_created = False
                # Create a new trainer instance
                st.session_state.trainer = YOLOTrainer()
                st.rerun()
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if not st.session_state.training_started:
            if st.button("Previous: Augmentation"):
                st.session_state.current_step = 4
                st.rerun()

# Render the appropriate step based on the current state
if st.session_state.current_step == 1:
    render_step_1()
elif st.session_state.current_step == 2:
    render_step_2()
elif st.session_state.current_step == 3:
    render_step_3()
elif st.session_state.current_step == 4:
    render_step_4()
elif st.session_state.current_step == 5:
    render_step_5()

# Footer
st.markdown("---")
st.markdown(
    "YOLO Trainer | Built with Streamlit & Ultralytics YOLOv8"
) 