# LLM From Scratch

An educational project implementing a Generative Pre-trained Transformer (GPT) language model from scratch using PyTorch. The goal of this project is to build and understand the core components of modern Large Language Models step-by-step.

## Project Structure

The repository is organized around the sequential steps of building an LLM:

### Tokenization
- `simple_tokenizer.py`: A custom character/word-level tokenizer built from scratch.
- `custom_tokenization_demo.py`: Demonstration of building a vocabulary and tokenizing text using the custom tokenizer.
- `bpe_tokenization_demo.py`: Demonstration of Byte Pair Encoding (BPE) using OpenAI's `tiktoken` library (GPT-2 tokenizer).

### Data Preparation
- `dataset.py`: PyTorch `Dataset` and `DataLoader` implementation for generating input-target pairs using a sliding context window.
- `the_verdict.txt`: A small text corpus ("The Verdict" by Edith Wharton) used for testing tokenization and training.

### Embeddings
- `embeddings_demo.py`: Implementation combining Token Embeddings and Positional Embeddings into a unified input representation.

### Attention Mechanisms
- `attention-mechanisms.py`: Step-by-step implementations of attention:
  - `SelfAttention_v1`: Basic self-attention using raw matrix multiplications.
  - `SelfAttention_v2`: Self-attention using PyTorch `nn.Linear` layers.
  - `CausalAttention`: Self-attention with causal masking (to prevent looking at future tokens) and dropout.
- `simplified_attention.py`: A bare-bones, weightless attention mechanism for conceptual demonstration.

## Setup and Usage

1. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
   pip install tiktoken regex requests
   ```

3. Run any of the demo scripts to see the components in action:
   ```bash
   python embeddings_demo.py
   python bpe_tokenization_demo.py
   ```
