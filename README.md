# SemEval-2026 Task 2: EmoVA

This repository contains the model architecture for the SemEval-2026 Task 2: EmoVA (Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays) challenge.

## System Description

The system implements a regression-based neural network architecture designed to predict continuous emotional Valence and Arousal (VA) from longitudinal text data. The architecture is split into two primary configurations to address the different subtasks.

<img width="2816" height="1536" alt="Gemini_Generated_Image_7jks0y7jks0y7jks" src="https://github.com/user-attachments/assets/740fbcde-cae2-4f3d-9d3a-1df68fef604a" />


### Subtask 1 (Longitudinal Affect Assessment)
The primary model processes sequences of user essays to capture emotional variation over time. The pipeline consists of:

* **Text Encoder:** A Transformer-based language model extracts semantic features from the text. It supports Parameter-Efficient Fine-Tuning using BitFit, which unfreezes only bias terms for memory efficiency.
* **Set Attention Pooling:** Token embeddings are processed through an Induced Set Attention Block (ISAB) with 32 inducing points to reduce self-attention complexity. The output is compressed into 4 fixed-size document representations per text using Pooling by Multihead Attention (PMA).
* **Temporal Modeling:** A bidirectional LSTM encoder processes the sequence of document embeddings to capture the longitudinal dynamics of the user's emotional state. It uses `pack_padded_sequence` to dynamically skip padding tokens and improve computational efficiency.
* **Loss Functions:** The model is optimized using a combination of Masked Mean Squared Error (MSE) and Concordance Correlation Coefficient (CCC) Loss to maximize both absolute accuracy and relative sequence correlation.

### Subtask 2a (Forecasting)
The forecasting model extends the base architecture to predict future emotional state changes.

* **Multimodal Fusion:** The architecture concatenates the sequence of historical Valence and Arousal scores directly with the PMA-pooled text embeddings before passing them into the LSTM.
* **Delta Prediction:** The final hidden state of the LSTM is combined with the last known VA values and passed through a specialized prediction head to forecast the single future variation.

---

## Repository Structure

### `src/data/`
* **`__init__.py`**: Exports dataset and dataloader initialization functions.
* **`dataset.py`**: Contains `EmoVADataset` for loading complete user sequences and `EmoVADataset2a` for loading sliding-window historical text and VA score sequences.
* **`collate.py`**: Defines custom collation functions (`create_collate_fn` and `create_collate_fn_2a`) to handle batching, flattening, and padding of variable-length longitudinal text sequences and their targets.
* **`utils.py`**: Contains functions like `setup_dataloader` to initialize the tokenizer, dataset, and batch collator in a single pipeline.

### `src/models/`
* **`affect_model.py`**: Defines the main end-to-end PyTorch architectures (`AffectModel` and `AffectModel2a`).
* **`encoder.py`**: Wrapper for HuggingFace models implementing the frozen backbone and PEFT/BitFit configurations.
* **`set_attention.py`**: Implements `MAB`, `ISAB`, and `PMA` layers based on the Set Transformer architecture for sequence pooling.
* **`lstm.py`**: Contains the `LSTMEncoder` utilizing packed sequences to efficiently model temporal patterns while ignoring padding.
* **`heads.py`**: Defines the final `PredictionHead` for outputting VA coordinates.
* **`tokenizer_wrapper.py`**: Standardizes tokenization, padding, and truncation logic.

### `src/evaluation/`
* **`metrics.py`**: Implements official SemEval metrics, including between-user Pearson correlation, within-user Pearson correlation, and the composite score via Fisher's z-transformation. It also computes metrics for Subtask 2a forecasting.

### `src/training/`
* **`losses.py`**: Defines custom objective functions, including a masked Mean Squared Error (`masked_mse_loss`), a Concordance Correlation Coefficient loss (`ccc_loss`), and a combined function.
* **`trainer.py`**: Execution loop for Subtask 1, handling automatic mixed precision, gradient accumulation, model checkpointing, and evaluation against the composite correlation metric.
* **`trainer_2a.py`**: Parallel execution loop adapted for Subtask 2a, optimizing via standard MSE and evaluating via average Pearson correlation.
* **`utils.py`**: Provides core training utilities like `EarlyStopping` and gradient clipping.

### Root Directory
* **`utils.py`**: General utility functions for root execution scripts.
* **`eda.ipynb`**: Exploratory Data Analysis notebook.
* **`main.ipynb`**: Primary pipeline notebook for training and evaluating baseline models.
* **`SEMEVAL2026_EMOVA_ABLATION.ipynb`**: Execution notebook for isolating and evaluating specific architectural components on Subtask 1.
* **`SEMEVAL2026_EMOVA_ABLATION_2a.ipynb`**: Execution notebook for ablation studies specific to the Subtask 2a forecasting models.
* **`SEMEVAL2026_EMOVA_SUBMISSION.ipynb`**: Final inference notebook used to format predictions for Codabench submission.
* **`SemEval2026_EmoVA.pdf`**: System description paper detailing the architecture and experimental methodology.
