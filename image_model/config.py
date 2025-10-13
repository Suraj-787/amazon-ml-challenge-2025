"""
Configuration for Image-based Price Prediction with LoRA Fine-tuning
Optimized for H100 80GB GPU with 4-hour training constraint
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageModelConfig:
    """Configuration for image price prediction model"""
    
    # Model Architecture
    base_model: str = "openai/clip-vit-large-patch14"  # 304M params
    model_type: str = "clip"  # clip or vit
    
    # LoRA Configuration (Parameter-Efficient Fine-Tuning)
    use_lora: bool = True
    lora_r: int = 16  # Rank of LoRA adapters (8, 16, or 32)
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set in __post_init__
    
    # Regression Head
    hidden_dim: int = 512
    dropout: float = 0.2
    num_layers: int = 3  # Number of layers in regression head
    
    # Training Configuration (H100 Optimized)
    batch_size: int = 512  # Large batch for H100 80GB
    gradient_accumulation_steps: int = 1  # Not needed with large batch
    num_epochs: int = 10  # Train until validation loss plateaus
    learning_rate: float = 1e-4  # LoRA learning rate
    learning_rate_head: float = 1e-3  # Regression head learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Mixed Precision (H100 native)
    use_bf16: bool = True  # bfloat16 is native to H100
    use_torch_compile: bool = True  # Massive speedup on H100
    
    # Data Configuration
    image_size: int = 224  # CLIP default
    num_workers: int = 8  # Data loading workers
    prefetch_factor: int = 2
    
    # S3 Configuration
    s3_bucket: str = "amzn-ml"
    s3_train_prefix: str = "images/train/"
    s3_test_prefix: str = "images/test/"
    aws_region: str = "us-east-1"
    
    # Dataset Configuration
    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"
    train_limit: int = 35000  # Only 35k images available
    
    # Output Configuration
    output_dir: str = "image_model/output"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Early Stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Target Transformation
    use_log_transform: bool = True  # Predict log(price + 1)
    
    def __post_init__(self):
        """Set default LoRA target modules based on model type"""
        if self.lora_target_modules is None:
            if self.model_type == "clip":
                # Target attention layers in CLIP vision encoder
                self.lora_target_modules = ["q_proj", "v_proj"]
            else:
                # For ViT models
                self.lora_target_modules = ["query", "value"]
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


# Default configuration
CONFIG = ImageModelConfig()


def get_config(**kwargs) -> ImageModelConfig:
    """Get configuration with optional overrides"""
    config = ImageModelConfig()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
