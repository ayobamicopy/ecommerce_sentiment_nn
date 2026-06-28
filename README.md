# Deep Learning Sentiment Classifier: E-Commerce Experience Optimization
This repository contains an end-to-end NLP deep learning framework utilizing the **Discovery-to-Action (DTA)** blueprint to evaluate unstructured e-commerce text reviews and automate customer support routing workflows based on neural prediction confidence.

---

## 🛠️ Discovery Phase: Data Engineering & Cleaning

### Data Wrangling & Label Optimization
* **Missing Value Resolution:** Rows containing `null` or `NaN` values within the core text fields were dropped to maintain clear training tokens.
* **Rating Binarization:** Raw ordinal scale ratings were mapped into crisp, binary target markers:
  * **Ratings 4 to 5 Stars:** Configured as `1` (Positive Sentiment Profile)
  * **Ratings 1 to 2 Stars:** Configured as `0` (Negative Sentiment Profile)
* **Neutral Data Pruning:** All 3-star evaluations were intentionally filtered out. Removing ambiguous boundary elements sharpens the classification margin, preventing soft-gradient updates from confusing the model during backward propagation.

### Text Standardization & Preprocessing Vectorization
Text processing uses an integrated `TextVectorization` framework tracking these configurations:
* **Vocabulary Cap:** Structured at `10,000` unique tokens to map common words while minimizing memory overhead.
* **Standardization Protocol:** Forced down-casing and automatic punctuation stripping to ensure structural variants (e.g., "Broken!", "broken", "broken,") point to the exact same vocabulary index.
* **Sequence Calibration:** Sequences are padded or truncated to a fixed ceiling of `60` tokens to ensure stable array formatting during model ingestion.

---

## 🧠 Technical Phase: Neural Network Architecture

The architecture relies on a highly efficient, lightweight `Sequential` text-processing network structured as follows:[Raw String Input]
│
▼
[Keras TextVectorization Layer] (1D Int Sequences, Max Length = 60)
│
▼
[Embedding Layer] (Input Dim = 10,000, Output Dim = 16)
│
▼
[GlobalAveragePooling1D Layer] (Collapses sequence dimensions safely)
│
▼
[Dense Hidden Layer] (16 Units, ReLU Activation Function)
│
▼
[Dense Output Neuron] (1 Unit, Sigmoid Activation Function)
### Training Strategy & Compilation Settings
* **Loss Function:** `binary_crossentropy` (the standard for single-neuron probability mapping).
* **Optimization Algorithm:** `Adam` (configured with an adaptive base learning rate of `0.001`).
* **Evaluation Metrics:** Tracked via `binary_accuracy` across both training and validation groups over `10` training epochs to check for overfitting.

---

## 🚀 Action Phase: Business Workflow Integration

### Required Model Validation
When evaluated against the required verification phrase:
> *"The product arrived broken and I am very unhappy"*

The network correctly assigns an output probability value **approaching 0.00**, accurately marking it as a strong negative sentiment sample.

### Automated Support Routing Matrix
Instead of just classifying reviews as positive or negative, we treat the output score as an operational metric to drive automated business workflows:

| Output Score Range | Interpreted Class | Operational Action | Business Justification |
| :--- | :--- | :--- | :--- |
| **$\le$ 0.20** | Confirmed Negative | **Critical Escalation Queue** | High-confidence negative review; immediately sent to human agents for recovery to limit churn. |
| **0.21 to 0.79** | Ambiguous Neutral | **Human Triage Queue** | Low-confidence region; sent for manual inspection to capture nuances like sarcasm or mixed signals. |
| **$\ge$ 0.80** | Confirmed Positive | **Auto-Archive & Share** | High-confidence positive review; safely routed to marketing pipelines or loyalty rewards automation. |

### Operational Threshold Justification
Setting the critical negative escalation limit at `0.20` ensures high accuracy. By routing only reviews with a score of 0.20 or lower to our priority queue, we protect customer support agents from false alarms. This keeps response times fast for truly frustrated customers while routing more complex, mixed reviews to a dedicated triage system for manual review.

---

## ⚠️ Operational Deficiencies & Production Roadmap

1. **Vocabulary Gaps (Out-of-Vocabulary Tokens):** Rare slangs, specific product codes, or typo variants convert automatically into an `[UNK]` token, reducing context accuracy.
2. **Sarcasm Blind Spots:** Phrases like *"Incredible, it broke on day one"* look positive to word-averaging networks due to terms like "Incredible". Adding recurrent networks (LSTM) or attention layers (Transformers) will help track word order and catch context shifts.
3. **Sequence Aggregation Constraints:** The global average pooling step ignores word order, which can cause issues with long reviews that mix praise with criticism. Short sentence constraints must be enforced during active production inferences.
