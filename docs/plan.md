# Text Model Improvement Plan

Review of the baseline DeBERTa pipeline (~48–49% validation SMAPE) and the improvements to make next, ordered from most to least impactful. The outcome of this plan is summarised in [what-was-done.md](what-was-done.md).

## 1. Feature Engineering: The Single Biggest Improvement (High Impact)

`catalog_content` is not just a block of text; it's a semi-structured record with highly predictive fields. Treating it as a single string forces the DeBERTa model to learn how to parse text and understand concepts like "Pack of 6" or "Value: 72.0", which is very inefficient.

**Problem:** The model sees "Value: 72.0 Unit: Fl Oz" as just a sequence of tokens. The numerical value 72.0 is not being used as a number.

**Solution:** Extract structured features.

Modify the `DataPreprocessor` to parse these fields, using regular expressions or simple string splitting.

### Action Plan

Parse `catalog_content` and create new DataFrame columns for:

- `item_name` (str)
- `bullet_points` (str, concatenate all bullet points)
- `product_description` (str)
- `value` (float)
- `unit` (str, categorical)
- `pack_count` (int, extract from item name or bullet points, e.g. "Pack of 6")

```python
# In DataPreprocessor or a new function
def parse_catalog_content(text: str) -> dict:
    data = {}
    # Use regex to find "Value: 11.4"
    value_match = re.search(r"Value:\s*([\d\.]+)", text)
    data['value'] = float(value_match.group(1)) if value_match else np.nan

    # Find "Unit: Ounce"
    unit_match = re.search(r"Unit:\s*(\w+)", text)
    data['unit'] = unit_match.group(1).lower() if unit_match else 'unknown'

    # Find "Pack of 6"
    pack_match = re.search(r"\(Pack of (\d+)\)", text, re.IGNORECASE)
    data['pack_count'] = int(pack_match.group(1)) if pack_match else 1

    # Combine text fields for the transformer
    # This is a simplified example; you can get more sophisticated
    item_name_match = re.search(r"Item Name:\s*(.*?)\n", text)
    bullets_match = re.findall(r"Bullet Point \d+:\s*(.*?)\n", text)

    item_name = item_name_match.group(1) if item_name_match else ""
    bullets = " ".join(bullets_match)

    data['clean_text'] = item_name + " [SEP] " + bullets
    return data

# Apply this to your DataFrame
parsed_df = df['catalog_content'].apply(lambda x: pd.Series(parse_catalog_content(x)))
df = pd.concat([df, parsed_df], axis=1)
```

Then create a multimodal model that accepts both text embeddings and these new features:

- **Numerical features:** `value`, `pack_count`. Standardize these (e.g. with `StandardScaler`).
- **Categorical features:** `unit`. One-hot encode this feature.
- **Text features:** Use the new `clean_text` column as the input to DeBERTa.

## 2. Model Architecture and Training Strategy (High Impact)

Once there are structured features, the model needs to use them.

### Update `DeBERTaPricePredictor`

The `forward` method should accept `numerical_features` as an input, and concatenate the DeBERTa output (`cls_output`) with the processed numerical/categorical features before the regression head.

```python
# In DeBERTaPricePredictor class
class DeBERTaPricePredictor(nn.Module):
    def __init__(self, model_name, dropout, num_numerical_features):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size  # e.g., 768 for base

        # The input size of the first dense layer must be updated
        self.regression_head_input_size = self.hidden_size + num_numerical_features

        # Example head
        self.dense1 = nn.Linear(self.regression_head_input_size, 512)
        # ... rest of the head ...
        self.output = nn.Linear(256, 1)

    def forward(self, input_ids, attention_mask, numerical_features):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Concatenate text embeddings with numerical features
        combined_features = torch.cat([cls_output, numerical_features], dim=1)

        # Pass combined features through the head
        x = self.dense1(combined_features)
        # ... rest of the forward pass ...
        return output.squeeze(-1)
```

### Update `PriceDataset` and `DataLoader`

The `__getitem__` method must return the numerical features as a `torch.Tensor`, and the training loop must pass this extra tensor to the model.

### Use Mean Pooling

The `[CLS]` token represents the entire sequence, but averaging the last hidden states of all tokens (mean pooling) sometimes captures more nuanced information. It's a simple change to test and can often yield better results.

```python
# Alternative to cls_output in the forward pass
# outputs.last_hidden_state has shape [batch_size, seq_len, hidden_size]
mean_pooled_output = outputs.last_hidden_state.mean(axis=1)
combined_features = torch.cat([mean_pooled_output, numerical_features], dim=1)
```

### Implement k-Fold Cross-Validation

A single train/validation split is subject to random chance. 5-fold cross-validation gives a much more robust SMAPE estimate and generally leads to a better final submission: train 5 separate models and average their predictions on the test set.

## 3. Hyperparameter Tuning and Configuration (Medium Impact)

- **Model inconsistency:** `main.py` uses `microsoft/deberta-v3-base` while the documentation and other files refer to `deberta-v3-large`. `base` has 768 hidden dimensions and is much smaller than `large` (1024). Use the intended model consistently. `large` will likely perform better but needs more GPU memory and time; `base` is fine while implementing the feature engineering changes.
- **Learning rate:** The differential learning rates are good, but the head learning rate (1e-3) might be a bit high and could cause instability. Try tuning it in the range [1e-4, 5e-3].
- **Loss function:** MSE loss on the log-transformed price is a solid choice. L1 loss (mean absolute error) is a good alternative that is less sensitive to outliers:

  ```python
  self.l1_loss = nn.L1Loss()
  loss = self.l1_loss(predictions, prices)
  ```

- **Max sequence length:** `MAX_LENGTH` is 256. Analyse the distribution of token lengths in `clean_text` after parsing. If many descriptions are longer, information is being lost; if most are much shorter, decrease it to speed up training.

## 4. Code and Data Refinements (Low Impact)

- **Text cleaning:** Handle data-specific patterns, such as normalising units (e.g. 'ounce' and 'oz' become the same).
- **Outlier handling:** `validate_price` clips prices between $0.01 and $10,000. This could be data-driven instead, clipping at the 0.5 and 99.5 percentiles of the training price distribution.
- **Evaluation:** `calculate_price_range_metrics` in `evaluation.py` is useful for diagnosing where the model fails. Models often do poorly on very low- or very high-priced items; if one price range has a high SMAPE, consider separate models per price bucket.

## Prioritized Action Plan

1. **Feature engineering.** Stop treating `catalog_content` as a single text block.
   - Extract `value`, `unit` and `pack_count` as separate numerical/categorical features.
   - Create a clean, concatenated text input (item name + bullet points) for DeBERTa.
2. **Update the model.**
   - Modify `DeBERTaPricePredictor` and the data pipeline to handle multimodal input (text embeddings + numerical features).
   - Experiment with mean pooling instead of the `[CLS]` token output.
3. **Use cross-validation.** Switch from a single validation split to 5-fold cross-validation.
4. **Tune and iterate.**
   - Once the new pipeline works, tune hyperparameters such as learning rates and dropout.
   - Decide between `deberta-v3-base` and `deberta-v3-large` based on performance and compute.

Implementing the feature engineering and multimodal architecture alone should give a substantial improvement in SMAPE.
