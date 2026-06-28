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

The architecture relies on a highly efficient, lightweight `Sequential` text-processing network structured as follows:
