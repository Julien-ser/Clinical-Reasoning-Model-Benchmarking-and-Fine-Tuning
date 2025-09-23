## Clinical Reasoning Model Benchmarking and Fine-Tuning (MedCalc-Bench)

This repository benchmarks and fine-tunes Qwen3 models on MedCalc-Bench clinical calculation tasks. It covers prompt engineering (zero-shot, few-shot, CoT), optional advanced methods, parameter-efficient fine-tuning (LoRA/QLoRA), and category-wise evaluation.

### Key Artifacts
- `Clinical_Reasoning_Benchmarks.ipynb`: Main, end-to-end notebook (setup → inference → evaluation).
- `outputs/`: CSV outputs from batch inference for all model/prompt combinations.
- `pivot_results.csv`: Pivoted accuracy by category for each Model+Prompt.
- `overall_results.csv`: Overall accuracy summary.
- `aggregated_results.csv`: Row-wise predictions with parsed answers and in-range flags.
- `summary_statistics.csv`: Descriptive stats used in the write-up.
- `qwen_lora_*/`: LoRA/QLoRA adapters and checkpoints (if you fine-tune).

### Environment Setup
- Python 3.10+ recommended
- NVIDIA GPU suggested (T4/A10/RTX) for fast inference; CPU is supported but slow

Quick setup via the notebook (Colab/local):
```bash
pip install --upgrade pip
pip install "transformers>=4.43.0" accelerate peft bitsandbytes datasets evaluate scikit-learn seaborn matplotlib pandas numpy einops xformers
```
If you encounter CUDA/driver issues with `bitsandbytes`, you can skip it and load models in full precision at a memory cost.

### Running the Notebook
1. Open `Clinical_Reasoning_Benchmarks.ipynb`.
2. Run the setup cell to install dependencies.
3. Execute the data loading cell (dataset is pulled from the Hugging Face Hub).
4. Execute model loading and quick sanity generations (optional).
5. Run the batched inference cell to generate outputs for:
   - Models: Qwen3-0.6B, Qwen3-1.7B
   - Prompts: zero-shot, few-shot, CoT (and optional RAG variants if you enable them)
6. Aggregation cells will compute accuracy, produce pivot tables, and write artifacts to `outputs/` and project root.

Outputs will appear under `outputs/` as CSVs, and summary CSVs (`pivot_results.csv`, `overall_results.csv`, etc.) will be written at the repo root unless you change the paths.

### Reproducibility Notes
- Tokenization uses left-padding and left-truncation for decoder-only models to enable batching while preserving the most recent context.
- Generation parameters default to low variance for numeric/date extraction: `temperature=0.3`, `top_p=0.9`, and conservative `max_new_tokens`.
- CoT uses a larger `max_new_tokens` to allow brief reasoning while evaluation relies only on the extracted `Answer:` line.

### Regenerating Results
- Delete the corresponding CSVs in `outputs/` if you want to rerun a combination (the notebook skips files that already exist).
- Tweak `BATCH_SIZE` and `N_FEW_SHOT` depending on GPU VRAM and context length.
- For full reproducibility, pin package versions in a `requirements.txt` and export a precise CUDA/driver configuration.

### File Descriptions
- `outputs/*_improved.csv`: Row-level generations with raw outputs, extracted answers, parsed types, and `in_range` correctness.
- `pivot_results.csv`: Category-wise accuracy by Model+Prompt, ready for plotting and comparison.
- `overall_results.csv`: Overall mean accuracy per Model+Prompt.
- `aggregated_results.csv`: Combined long-form results across runs for further analysis.

### Inline Preview: pivot_results.csv
A compact preview of `pivot_results.csv` included here for quick inspection (values are accuracies in [0,1]):

| Model_Prompt | Overall | date | diagnosis | dosage | lab | physical | risk | severity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B_cot | 0.0813 | 0.0000 | 0.2333 | 0.0750 | 0.0734 | 0.1292 | 0.0333 | 0.0250 |
| 0.6B_few_shot | 0.0902 | 0.0000 | 0.2667 | 0.0750 | 0.0734 | 0.1583 | 0.0333 | 0.0250 |
| 0.6B_rag | 0.1668 | 0.0333 | 0.3833 | 0.0750 | 0.1131 | 0.4375 | 0.0625 | 0.0625 |
| 0.6B_zero_shot | 0.0609 | 0.0000 | 0.1500 | 0.0750 | 0.0765 | 0.0917 | 0.0208 | 0.0125 |
| 0.6B_zero_shot_quantized_lora | 0.0104 | 0.0000 | 0.0000 | 0.0250 | 0.0061 | 0.0417 | 0.0000 | 0.0000 |
| 1.7B_cot | 0.1379 | 0.0000 | 0.2833 | 0.0750 | 0.1193 | 0.2167 | 0.1208 | 0.1500 |
| 1.7B_few_shot | 0.1061 | 0.0167 | 0.1833 | 0.0500 | 0.1468 | 0.1333 | 0.1000 | 0.1125 |
| 1.7B_rag | 0.1759 | 0.0333 | 0.2833 | 0.0750 | 0.1437 | 0.4625 | 0.1083 | 0.1250 |
| 1.7B_zero_shot | 0.1234 | 0.0000 | 0.2333 | 0.0750 | 0.1468 | 0.1583 | 0.1000 | 0.1500 |
| 1.7B_zero_shot_quantized_lora | 0.0454 | 0.0000 | 0.0667 | 0.0000 | 0.0550 | 0.0792 | 0.0792 | 0.0375 |

Note: These are sample values from the provided CSV; your results may vary with different seeds, hardware, or dependency versions.

### Visualization
You can load `pivot_results.csv` in Python to create concise plots:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pivot = pd.read_csv("pivot_results.csv")
long = pivot.melt(id_vars=["Model_Prompt"], var_name="category", value_name="accuracy")
plt.figure(figsize=(10,5))
sns.barplot(data=long[long["category"]!="Overall"], x="category", y="accuracy", hue="Model_Prompt")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

### Citation
Qwen3 Models
Yang, A., Liu, Y., Guo, Z., et al. “Qwen3: Technical Report.” arXiv preprint arXiv:2505.09388, 2025.
Cite as:
Yang, A., Liu, Y., Guo, Z., et al. Qwen3: Technical Report. arXiv:2505.09388 [cs.CL], 2025.

MedCalc-Bench Dataset
Khandekar, N., Jin, Q., Xiong, G., Dunn, S., Applebaum, S., et al. “MedCalc-Bench: Evaluating Large Language Models for Medical Calculations.” arXiv preprint arXiv:2406.12036, 2024.
Cite as:
Khandekar, N., Jin, Q., Xiong, G., et al. MedCalc-Bench: Evaluating Large Language Models for Medical Calculations. arXiv:2406.12036 [cs.CL], 2024.
