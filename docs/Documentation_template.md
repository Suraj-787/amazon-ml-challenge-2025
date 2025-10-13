# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** [AI Brewery]  
**Team Members:** [Pavan J, Ayush Shankar Prasad, Arshan Ali Khan, Aditya Agarwal]  
**Submission Date:** October 2025

---

## 1. Executive Summary

We developed a multimodal fusion architecture that combines textual product descriptions with visual product images to predict pricing with high accuracy. Our approach leverages advanced text preprocessing, CLIP-based image feature extraction, and an ensemble of XGBoost and LightGBM models, achieving a validation SMAPE of 57.51%. The key innovation lies in successfully integrating both textual and visual information to capture comprehensive product characteristics that influence pricing.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
*Describe how you interpreted the pricing challenge and key insights discovered during EDA.*

**Key Observations:**
- Product pricing is influenced by both textual descriptions (brand, quantity, quality indicators) and visual appearance
- Text features alone achieved ~60% SMAPE, indicating room for improvement through multimodal learning
- Price distribution shows significant variance ($0-$1000+), requiring robust models that handle diverse price ranges
- Image links are available for all products, providing rich visual information for price prediction
- Text preprocessing revealed important patterns: quantity indicators, brand names, quality descriptors significantly impact pricing

### 2.2 Solution Strategy
*Outline your high-level approach (e.g., multimodal learning, ensemble methods, etc.)*

**Approach Type:** Multimodal Ensemble Learning  
**Core Innovation:** Integration of CLIP-based visual features with comprehensive text feature engineering for product price prediction

---

## 3. Model Architecture

### 3.1 Architecture Overview
*Describe your model architecture with a simple diagram or flowchart if possible.*

```
Input Data → Text Processing → Text Features (3,041 dim)
     ↓                              ↓
Product Data → Image Processing → Image Features (512 dim)
     ↓                              ↓
     └─────────── Feature Fusion ────────────┘
                        ↓
              Multimodal Features (3,553 dim)
                        ↓
              XGBoost + LightGBM Ensemble
                        ↓
                   Price Predictions
```

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: HTML removal, lowercase conversion, special character cleaning, stop word removal
- [x] Model type: TF-IDF vectorization with advanced feature engineering
- [x] Key parameters: 
  - TF-IDF: max_features=3000, ngram_range=(1,2)
  - Quantity extraction: oz, lb, ml, fl_oz, g, kg, pack, count
  - Keyword flags: organic, premium, natural, gluten_free, non_gmo, imported, artisan, gourmet, luxury, deluxe, combo, bundle, variety, assorted, multi
  - Text statistics: character count, word count, sentence count, bullet point count
  - Brand name extraction and encoding

**Image Processing Pipeline:**
- [x] Preprocessing steps: Image download, resize to 224x224, RGB conversion
- [x] Model type: OpenAI CLIP (Vision Transformer Base Patch 32)
- [x] Key parameters: 
  - Model: openai/clip-vit-base-patch32
  - Output dimension: 512 features per image
  - Batch processing: 128 images per batch
  - Parallel downloads: 200 concurrent threads

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 57.51% (Ensemble Validation)
- **Other Metrics:** 
  - XGBoost Validation SMAPE: 57.96%
  - LightGBM Validation SMAPE: 60.07%
  - Ensemble Train SMAPE: 35.28%
  - Feature Dimensions: 3,553 (3,041 text + 512 image)

### 4.2 Individual Model Performance
| Model | Train SMAPE | Validation SMAPE |
|-------|-------------|-----------------|
| XGBoost | 24.94% | 57.96% |
| LightGBM | 46.68% | 60.07% |
| **Ensemble** | **35.28%** | **57.51%** |

---

## 5. Conclusion

Our multimodal fusion approach successfully combines textual and visual product information to achieve competitive price prediction performance. The integration of CLIP-based image features with comprehensive text preprocessing created a robust 3,553-dimensional feature space that captures both semantic and visual product characteristics. The ensemble of XGBoost and LightGBM models provides stable predictions across diverse price ranges, demonstrating the effectiveness of multimodal learning for e-commerce pricing challenges.

---

## Appendix

### A. Code artefacts
*Include drive link for your complete code directory*

**Main Implementation Files:**
- `multimodal_fusion.py` - Complete multimodal fusion architecture
- `ultra_fast_cpu_multimodal.py` - Optimized image feature extraction
- `multimodal_predictions.csv` - Final predictions (75,000 samples)
- `multimodal_xgboost_model.pkl` - Trained XGBoost model
- `multimodal_lightgbm_model.pkl` - Trained LightGBM model

### B. Additional Results
*Include any additional charts, graphs, or detailed results*

**Technical Specifications:**
- Training samples: 75,000
- Test predictions: 75,000
- Feature extraction time: ~20 minutes (CPU optimized)
- Model training time: ~10 minutes
- Total feature dimensions: 3,553
- Prediction range: $0.00 - $764.62
- Mean prediction: $23.87

**Performance Improvements:**
- Baseline text-only SMAPE: ~77%
- Advanced text features SMAPE: ~60%
- Multimodal fusion SMAPE: 57.51%
- Overall improvement: ~19.5% reduction in SMAPE

---

**Note:** This multimodal fusion approach represents a significant advancement in product price prediction by successfully integrating both textual and visual information sources.