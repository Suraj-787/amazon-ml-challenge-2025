#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly
Run this before starting training to catch issues early
"""

import sys
import os

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_check(name, status, details=""):
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{symbol}{reset} {name}")
    if details:
        print(f"  → {details}")

def check_python():
    """Check Python version"""
    print_header("Python Environment")
    
    version = sys.version_info
    is_ok = version.major == 3 and version.minor >= 8
    print_check(
        "Python Version",
        is_ok,
        f"Python {version.major}.{version.minor}.{version.micro}"
    )
    return is_ok

def check_pytorch():
    """Check PyTorch installation"""
    print_header("PyTorch")
    
    try:
        import torch
        print_check("PyTorch Installed", True, f"Version {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print_check("CUDA Available", cuda_available)
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_check("GPU", True, f"{gpu_name} ({gpu_memory:.1f} GB)")
            
            # Check if H100
            is_h100 = "H100" in gpu_name
            print_check("H100 GPU", is_h100, "Optimized for H100" if is_h100 else "Will work but slower")
            
            # Check memory
            has_enough_memory = gpu_memory >= 40
            print_check(
                "GPU Memory",
                has_enough_memory,
                f"{gpu_memory:.1f} GB {'(sufficient)' if has_enough_memory else '(may need to reduce batch size)'}"
            )
        
        return cuda_available
    except ImportError:
        print_check("PyTorch Installed", False, "pip install torch")
        return False

def check_dependencies():
    """Check required dependencies"""
    print_header("Dependencies")
    
    deps = {
        'transformers': 'Hugging Face Transformers',
        'peft': 'Parameter-Efficient Fine-Tuning',
        'boto3': 'AWS SDK',
        'pandas': 'Data Processing',
        'numpy': 'Numerical Computing',
        'PIL': 'Image Processing',
        'tqdm': 'Progress Bars',
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            if module == 'PIL':
                import PIL
                version = PIL.__version__
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
            print_check(name, True, f"Version {version}")
        except ImportError:
            print_check(name, False, f"pip install {module}")
            all_ok = False
    
    return all_ok

def check_aws():
    """Check AWS credentials and S3 access"""
    print_header("AWS Configuration")
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Check credentials
        try:
            s3 = boto3.client('s3', region_name='us-east-1')
            print_check("AWS Credentials", True)
            
            # Check S3 bucket access
            try:
                s3.head_bucket(Bucket='amzn-ml')
                print_check("S3 Bucket Access", True, "s3://amzn-ml")
                
                # Check if images exist
                try:
                    response = s3.list_objects_v2(
                        Bucket='amzn-ml',
                        Prefix='images/train/',
                        MaxKeys=10
                    )
                    count = response.get('KeyCount', 0)
                    print_check(
                        "Training Images",
                        count > 0,
                        f"Found {count} sample images"
                    )
                except Exception as e:
                    print_check("Training Images", False, str(e))
                
                return True
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    print_check("S3 Bucket Access", False, "Bucket not found")
                elif error_code == '403':
                    print_check("S3 Bucket Access", False, "Access denied")
                else:
                    print_check("S3 Bucket Access", False, str(e))
                return False
        except NoCredentialsError:
            print_check("AWS Credentials", False, "Configure AWS credentials")
            return False
    except ImportError:
        print_check("Boto3", False, "pip install boto3")
        return False

def check_data_files():
    """Check if data files exist"""
    print_header("Data Files")
    
    files = {
        'data/train.csv': 'Training Data',
        'data/test.csv': 'Test Data',
    }
    
    all_ok = True
    for path, name in files.items():
        exists = os.path.exists(path)
        if exists:
            size = os.path.getsize(path) / 1e6  # MB
            print_check(name, True, f"{path} ({size:.1f} MB)")
        else:
            print_check(name, False, f"{path} not found")
            all_ok = False
    
    return all_ok

def check_output_dir():
    """Check output directory"""
    print_header("Output Directory")
    
    output_dir = 'image_model/output'
    exists = os.path.exists(output_dir)
    
    if not exists:
        try:
            os.makedirs(output_dir, exist_ok=True)
            print_check("Output Directory", True, f"Created {output_dir}")
        except Exception as e:
            print_check("Output Directory", False, str(e))
            return False
    else:
        print_check("Output Directory", True, output_dir)
    
    # Check write permissions
    test_file = os.path.join(output_dir, '.test')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print_check("Write Permissions", True)
        return True
    except Exception as e:
        print_check("Write Permissions", False, str(e))
        return False

def check_model_config():
    """Check model configuration"""
    print_header("Model Configuration")
    
    try:
        from image_model.config import CONFIG
        
        print_check("Config Loaded", True)
        print(f"  → Base Model: {CONFIG.base_model}")
        print(f"  → LoRA Rank: {CONFIG.lora_r}")
        print(f"  → Batch Size: {CONFIG.batch_size}")
        print(f"  → Epochs: {CONFIG.num_epochs}")
        print(f"  → Use BF16: {CONFIG.use_bf16}")
        print(f"  → Use torch.compile: {CONFIG.use_torch_compile}")
        
        # Check if batch size is reasonable
        if CONFIG.batch_size > 512:
            print_check(
                "Batch Size",
                False,
                f"{CONFIG.batch_size} may be too large, try 512 or less"
            )
        else:
            print_check("Batch Size", True, f"{CONFIG.batch_size}")
        
        return True
    except Exception as e:
        print_check("Config Loaded", False, str(e))
        return False

def estimate_training_time():
    """Estimate training time"""
    print_header("Training Estimates")
    
    try:
        import torch
        from image_model.config import CONFIG
        
        # Rough estimates
        samples = 35000
        batch_size = CONFIG.batch_size
        epochs = CONFIG.num_epochs
        
        steps_per_epoch = samples // batch_size
        total_steps = steps_per_epoch * epochs
        
        # Time per step depends on GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "H100" in gpu_name:
                time_per_step = 0.3  # seconds
            elif "A100" in gpu_name:
                time_per_step = 0.5
            else:
                time_per_step = 1.0
        else:
            time_per_step = 10.0  # CPU (very slow)
        
        total_time = total_steps * time_per_step / 3600  # hours
        
        print(f"  → Training Samples: {samples:,}")
        print(f"  → Batch Size: {batch_size}")
        print(f"  → Steps per Epoch: {steps_per_epoch}")
        print(f"  → Total Epochs: {epochs}")
        print(f"  → Total Steps: {total_steps:,}")
        print(f"  → Estimated Time: {total_time:.1f} hours")
        
        if total_time > 4:
            print_check(
                "Training Time",
                False,
                f"{total_time:.1f} hours exceeds 4-hour target"
            )
        else:
            print_check(
                "Training Time",
                True,
                f"{total_time:.1f} hours (within 4-hour target)"
            )
        
        return True
    except Exception as e:
        print(f"  → Could not estimate: {e}")
        return False

def main():
    """Run all checks"""
    print("\n" + "🔍 Image Model Setup Verification" + "\n")
    
    checks = [
        ("Python", check_python),
        ("PyTorch", check_pytorch),
        ("Dependencies", check_dependencies),
        ("AWS", check_aws),
        ("Data Files", check_data_files),
        ("Output Directory", check_output_dir),
        ("Model Config", check_model_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error checking {name}: {e}")
            results.append((name, False))
    
    # Estimate training time
    estimate_training_time()
    
    # Summary
    print_header("Summary")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print("\n✅ All checks passed! Ready to train.")
        print("\nRun: ./image_model/run_training.sh")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFailed checks:")
        for name, result in results:
            if not result:
                print(f"  - {name}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
