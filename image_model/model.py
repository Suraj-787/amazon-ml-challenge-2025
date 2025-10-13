"""
LoRA-based Image Price Prediction Model
Uses CLIP vision encoder with LoRA adapters for parameter-efficient fine-tuning
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class ImagePricePredictor(nn.Module):
    """
    Image-based price prediction model with LoRA fine-tuning
    
    Architecture:
    1. Frozen CLIP Vision Encoder (304M params) - pre-trained knowledge
    2. LoRA Adapters (few M params) - trainable, injected into attention layers
    3. Regression Head (few M params) - trainable, maps embeddings to price
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        logger.info(f"Loading base model: {config.base_model}")
        
        # Load pre-trained CLIP vision model
        self.vision_model = CLIPVisionModel.from_pretrained(config.base_model)
        
        # Get embedding dimension
        self.embed_dim = self.vision_model.config.hidden_size  # 1024 for CLIP-Large
        
        if config.use_lora:
            logger.info("Applying LoRA adapters...")
            self._apply_lora()
        else:
            # Freeze all vision model parameters
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # Build regression head
        self.regression_head = self._build_regression_head()
        
        logger.info(f"Model initialized with {self.count_parameters()} total parameters")
        logger.info(f"Trainable parameters: {self.count_trainable_parameters()}")
    
    def _apply_lora(self):
        """Apply LoRA adapters to vision model"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        # Apply LoRA to vision model
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        
        logger.info(f"LoRA Config: r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        logger.info(f"Target modules: {self.config.lora_target_modules}")
        
        # Print trainable parameters
        self.vision_model.print_trainable_parameters()
    
    def _build_regression_head(self):
        """Build multi-layer regression head"""
        layers = []
        
        input_dim = self.embed_dim
        hidden_dim = self.config.hidden_dim
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        ])
        
        # Hidden layers
        for _ in range(self.config.num_layers - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, pixel_values):
        """
        Forward pass
        
        Args:
            pixel_values: Tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            predictions: Tensor of shape (batch_size, 1)
        """
        # Get image embeddings from CLIP
        outputs = self.vision_model(pixel_values=pixel_values)
        
        # Use pooled output (CLS token equivalent)
        image_embeds = outputs.pooler_output  # (batch_size, embed_dim)
        
        # Pass through regression head
        predictions = self.regression_head(image_embeds)
        
        return predictions
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_lora_weights(self, output_dir: str):
        """Save only LoRA adapter weights (very small file)"""
        if self.config.use_lora:
            self.vision_model.save_pretrained(output_dir)
            logger.info(f"LoRA weights saved to {output_dir}")
    
    def save_regression_head(self, output_path: str):
        """Save regression head weights"""
        torch.save(self.regression_head.state_dict(), output_path)
        logger.info(f"Regression head saved to {output_path}")
    
    def load_lora_weights(self, lora_dir: str):
        """Load LoRA adapter weights"""
        if self.config.use_lora:
            from peft import PeftModel
            self.vision_model = PeftModel.from_pretrained(
                self.vision_model.base_model,
                lora_dir
            )
            logger.info(f"LoRA weights loaded from {lora_dir}")
    
    def load_regression_head(self, head_path: str):
        """Load regression head weights"""
        self.regression_head.load_state_dict(torch.load(head_path))
        logger.info(f"Regression head loaded from {head_path}")


def create_model(config):
    """Factory function to create model"""
    model = ImagePricePredictor(config)
    
    # Apply torch.compile for H100 speedup
    if config.use_torch_compile:
        logger.info("Applying torch.compile() for H100 optimization...")
        try:
            model = torch.compile(model, mode='max-autotune')
            logger.info("torch.compile() applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
    
    return model
