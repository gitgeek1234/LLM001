# Text-to-SQL Fine-Tuning with LoRA (Qwen2.5-0.5B)

Fine-tuned a small language model to generate SQL queries from natural language questions, given a table schema — using parameter-efficient fine-tuning (LoRA) to make it feasible on consumer hardware.

## Task
Given a database schema and a natural language question, generate the corresponding SQL query.

**Example:**

    Schema: CREATE TABLE table_name_50 (venue VARCHAR, away_team VARCHAR)
    Question: When Essendon played away, where did they play?
    Generated SQL: SELECT venue FROM table_name_50 WHERE away_team = "essendon"

## Setup
- **Base model:** Qwen2.5-0.5B (Alibaba) — not trained from scratch; pretrained weights used as-is and adapted via fine-tuning
- **Method:** LoRA (Low-Rank Adaptation) via Hugging Face `peft` — only ~0.1-1% of total parameters are trained, rest of the base model stays frozen and quantized to 4-bit
- **Dataset:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context) (~78K schema/question/SQL examples)
- **Hardware:** Google Colab, T4 GPU (16GB VRAM), 4-bit quantization via `bitsandbytes`
- **Training:** 1 epoch, LoRA rank 8, targeting attention query/value projections, learning rate 2e-4

## Results
- **Exact-match accuracy (held-out eval, n=200): 21%**
- This number is conservative, not a full picture — manual inspection of failures shows:
  - A meaningful fraction are **metric artifacts**: the model produced a semantically correct query that differs only in formatting (e.g. `"7-1"` vs `"7–1"`, column name casing, minor syntax variants that exact-string-match penalizes unfairly)
  - The model performs noticeably better on **single-table queries** than **multi-table JOIN queries**, where it sometimes hallucinates table/column names — consistent with the limited capacity of a 0.5B model and only 1 training epoch
- Training loss dropped cleanly from ~1.8 to ~0.93-0.97 and stabilized, indicating the model converged rather than failing to learn

## Known Limitations
- Exact-match is a strict metric; a looser semantic/execution-based match would likely show meaningfully higher real-world accuracy
- Multi-table JOIN reasoning is the clearest weak point
- Only trained for 1 epoch due to time constraints — loss had not fully plateaued

## Next Steps (post-submission)
- Train for 2-3 epochs (loss was still improving)
- Expand LoRA to rank 16 and additional attention modules (`k_proj`, `o_proj`) for more adaptation capacity
- Check whether `max_length=512` truncates longer multi-table schemas, which could directly explain JOIN-query failures
- Add execution-match evaluation (actually running generated SQL against a test DB) as a stronger, less format-sensitive metric
- Check class balance of JOIN vs. single-table examples in training data; oversample JOINs if underrepresented

## Repo Contents
- `notebook.ipynb` — full training and evaluation pipeline
- `adapter/` — saved LoRA adapter weights (load on top of the base Qwen2.5-0.5B model via `peft`)

## Reproducing
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, "./adapter")
tokenizer = AutoTokenizer.from_pretrained("./adapter")
```
