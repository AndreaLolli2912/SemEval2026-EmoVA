# SemEval 2026 Task 2: EmoVA

Longitudinal Affect Assessment from Text

## Task

Given a sequence of texts from a user over time, predict valence and arousal scores for each document.

## Installation

```bash
git clone https://github.com/AndreaLolli2912/SemEval2026-EmoVA.git
cd SemEval2026-EmoVA
```

## Project Structure

```
├── src/
│   ├── data/           # Dataset and collation
│   ├── models/         # BERT + ISAB + PMA + LSTM architecture
│   ├── training/       # Training loop and losses
│   └── evaluation/     # Official SemEval metrics
```

## Architecture

BERT → ISAB → PMA → BiLSTM → Prediction Head
