"""
S3-based Image Dataset for streaming images directly from S3
Avoids disk bottlenecks by loading images directly into GPU memory
"""

import io
import logging
import boto3
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Optional, Tuple
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3ImageDataset(Dataset):
    """
    PyTorch Dataset that streams images directly from S3
    Optimized for high-throughput training on H100
    """
    
    def __init__(
        self,
        csv_path: str,
        s3_bucket: str,
        s3_prefix: str,
        image_size: int = 224,
        use_log_transform: bool = True,
        aws_region: str = "us-east-1",
        limit: Optional[int] = None,
        augment: bool = False,
    ):
        """
        Args:
            csv_path: Path to CSV with sample_id, image_link, price columns
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for images (e.g., "images/train/")
            image_size: Target image size
            use_log_transform: Whether to log-transform prices
            aws_region: AWS region
            limit: Limit number of samples (for testing)
            augment: Whether to apply data augmentation
        """
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.image_size = image_size
        self.use_log_transform = use_log_transform
        self.augment = augment
        
        # Initialize S3 client
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Load CSV data
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
            logger.info(f"Limited dataset to {limit} samples")
        
        # Filter to only samples with images (35k out of 75k)
        self.sample_ids = df['sample_id'].astype(str).tolist()
        
        # For training data, we have prices
        if 'price' in df.columns:
            self.prices = df['price'].values
            if self.use_log_transform:
                self.prices = np.log1p(self.prices)  # log(price + 1)
        else:
            self.prices = None
        
        logger.info(f"Loaded {len(self.sample_ids)} samples")
        
        # Define image transforms
        self.transform = self._get_transforms(augment)
        
        # Cache for failed loads
        self.failed_loads = set()
    
    def _get_transforms(self, augment: bool):
        """Get image transformation pipeline"""
        if augment:
            # Training augmentation
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
        else:
            # Validation/test transforms
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
    
    def _load_image_from_s3(self, sample_id: str) -> Optional[Image.Image]:
        """Load image from S3"""
        # Try common image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        for ext in extensions:
            s3_key = f"{self.s3_prefix}{sample_id}{ext}"
            
            try:
                response = self.s3_client.get_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key
                )
                image_data = response['Body'].read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                return image
            
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'NoSuchKey':
                    continue  # Try next extension
                else:
                    logger.warning(f"Error loading {s3_key}: {e}")
                    continue
            
            except Exception as e:
                logger.warning(f"Error processing {s3_key}: {e}")
                continue
        
        return None
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image: Tensor of shape (3, image_size, image_size)
            price: Tensor of shape (1,) - log-transformed if use_log_transform=True
            sample_id: String identifier
        """
        sample_id = self.sample_ids[idx]
        
        # Skip if previously failed
        if sample_id in self.failed_loads:
            # Return a black image as placeholder
            image = torch.zeros(3, self.image_size, self.image_size)
            price = torch.tensor([0.0]) if self.prices is not None else torch.tensor([0.0])
            return image, price, sample_id
        
        # Load image from S3
        image = self._load_image_from_s3(sample_id)
        
        if image is None:
            logger.warning(f"Failed to load image for sample {sample_id}")
            self.failed_loads.add(sample_id)
            # Return black image as placeholder
            image = torch.zeros(3, self.image_size, self.image_size)
            price = torch.tensor([0.0]) if self.prices is not None else torch.tensor([0.0])
            return image, price, sample_id
        
        # Apply transforms
        try:
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error transforming image {sample_id}: {e}")
            self.failed_loads.add(sample_id)
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Get price
        if self.prices is not None:
            price = torch.tensor([self.prices[idx]], dtype=torch.float32)
        else:
            price = torch.tensor([0.0], dtype=torch.float32)
        
        return image, price, sample_id


def collate_fn(batch):
    """Custom collate function to handle failed image loads"""
    images, prices, sample_ids = zip(*batch)
    
    # Stack tensors
    images = torch.stack(images)
    prices = torch.stack(prices)
    
    return images, prices, sample_ids
