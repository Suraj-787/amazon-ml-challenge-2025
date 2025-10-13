# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Monk Builders  
**Team Members:** [Suraj Kumar,Himanshu Rai,Pratik Gautam,Impana KS]
**Submission Date:** October 13, 2025

---

## 1. Executive Summary
Our solution leverages DeBERTa-V3-Large transformer model for smart product pricing, achieving a SMAPE score of 49.23%. We implemented a multimodal approach with advanced text preprocessing and a custom regression head designed specifically for price prediction tasks.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
We interpreted this as a regression problem where product catalog content (text descriptions) must be mapped to accurate price predictions. Our EDA revealed significant price variance across product categories and the importance of text quality in price determination.

**Key Observations:**
- Price distribution follows log-normal pattern requiring log-space transformations
- Text length and quality strongly correlate with price accuracy
- Product descriptions contain rich semantic information valuable for pricing

### 2.2 Solution Strategy
We implemented a transformer-based approach using DeBERTa-V3-Large with specialized components for price prediction.

**Approach Type:** Single Model with Advanced Architecture  
**Core Innovation:** Custom multi-layer regression head with differential learning rates for encoder vs. head components

---

## 3. Model Architecture

### 3.1 Architecture Overview
Our model consists of:
1. DeBERTa-V3-Large encoder (304M parameters) for text understanding
2. Multi-layer regression head for price prediction
3. Log-space transformations for improved price modeling
4. Custom SMAPE loss function for direct optimization

### 3.2 Model Components

**Text Processing Pipeline:**
- [x] Preprocessing steps: HTML removal, unicode normalization, whitespace cleaning
- [x] Model type: DeBERTa-V3-Large transformer
- [x] Key parameters: Max length 256, dropout 0.1

**Price Processing Pipeline:**
- [x] Preprocessing steps: Log transformation, outlier filtering
- [x] Model type: Multi-layer regression head (1024→512→256→1)
- [x] Key parameters: GELU activation, differential learning rates

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 49.23%
- **Other Metrics:** Comprehensive evaluation including MAE, RMSE, R² metrics across price ranges

## 5. Conclusion
Our DeBERTa-V3-Large based approach successfully captures semantic relationships between product descriptions and pricing, achieving competitive performance through careful architecture design and preprocessing strategies.
