# LLM From Scratch

An educational project implementing a Generative Pre-trained Transformer (GPT) language model from scratch using PyTorch. The goal of this project is to build and understand the core components of modern Large Language Models step-by-step.

## Technologies Used
- **Python 3.11**
- **PyTorch** (with CUDA 13.0 for GPU acceleration)
- **tiktoken** (OpenAI's Byte Pair Encoding tokenizer)

## Project Structure

The repository is organized into core components (`src/`) and testing/learning scripts (`demo/`):

### `src/` (Core Architecture)
- `dataset.py`: PyTorch `Dataset` and `DataLoader` implementation for generating input-target pairs using a sliding context window.
- `MultiHeadedAttention.py`: Highly optimized, fully vectorized Multi-Head Attention layer.

### `demo/` (Learning & Testing)
- `simple_tokenizer.py`: A custom character/word-level tokenizer built from scratch.
- `custom_tokenization_demo.py`: Demonstration of building a vocabulary and tokenizing text using the custom tokenizer.
- `bpe_tokenization_demo.py`: Demonstration of Byte Pair Encoding (BPE) using OpenAI's `tiktoken` library (GPT-2 tokenizer).
- `embeddings_demo.py`: Implementation combining Token Embeddings and Positional Embeddings into a unified input representation.
- `attention_mechanisms.py`: Step-by-step implementations of attention (Basic, PyTorch Linear, and Causal Masking).
- `simplified_attention.py`: A bare-bones, weightless attention mechanism for conceptual demonstration.

### Data
- `the_verdict.txt`: A small text corpus ("The Verdict" by Edith Wharton) used for testing tokenization and training.

## Setup and Usage

1. Create a Python 3.11 virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies (including PyTorch with CUDA support):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install tiktoken regex requests
   ```

3. Run any of the demo scripts to see the components in action:
   ```bash
   python demo/embeddings_demo.py
   python demo/bpe_tokenization_demo.py
   ```
