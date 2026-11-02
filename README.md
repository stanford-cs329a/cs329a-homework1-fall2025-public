# CS329a Self-Improving AI Agents - Homework 1 (Fall 2025)

Enhanced homework assignment covering test-time compute scaling techniques for self-improving AI agents.

## Overview

This homework explores various techniques to improve model performance during inference, including:

1. Zero-shot Evaluation (10 points)
2. Majority Voting (30 points)
3. Best-of-N with a Generative Reward Model (30 points)
4. Self-Improvement with Feedback (45 points)
5. Analysis (15 points)

## Setup

```bash
1. Download anaconda from https://www.anaconda.com/ and install
2. Create a new environment, and install the repo:

```bash
conda create -n cs329a-hw1 python=3.10 -y
conda activate cs329a-hw1
pip install -e .  # Run this from root of the repo.
```

This will make sure the package is installed with requirements, so you can import functionality from `cs329_hw1`. For instance, you can use the `get_sampler` function to get a sampler.

**Set up API Keys:**

Using .env file:
1. Copy `.env.example` to `.env`
2. Edit `.env` and add your API keys:
   - `TOGETHER_API_KEY`: Your Together AI API key
   - `HF_TOKEN`: Your Hugging Face token (get one at https://huggingface.co/settings/tokens)
3. In your notebook, this will set your env variables properly:
```python
from dotenv import load_dotenv
load_dotenv()
```
### Troubleshooting

**Kernel Issues**: If the notebook isn't recognizing your conda environment, run:
```bash
conda activate cs329a-hw1
conda install ipykernel -y
python -m ipykernel install --user --name cs329a-hw1 --display-name "cs329a-hw1"
```

**Package Import Errors**: If you get `ModuleNotFoundError` (e.g., for `tenacity`), ensure packages are installed in the correct environment:
```bash
conda activate cs329a-hw1
python -m pip install -r requirements.txt
```

## Usage

```python
from cs329_hw1.tasks import AIME25
from cs329_hw1.methods import get_sampler, get_verifier

aime25 = AIME25()
problems = aime25.get_problems()
system_prompt = aime25.get_system_prompt()
verifier = get_verifier("aime25")

# Example: Sample with Together AI using Qwen model
method = get_sampler("sample_multiple", "together_ai/Qwen/Qwen3.5-9B", temperature=0.7, n_samples=16, system_prompt=system_prompt)
```

## Features

- **AIME25 Dataset**: Challenging mathematical problems from the American Invitational Mathematics Examination
- **Multiple Sampling Methods**: Greedy, multiple sampling, majority voting, LLM voting
- **Self-Improvement System**: RLEF (Reinforcement Learning from Execution Feedback) with critique and regeneration
- **Comprehensive Analysis**: Detailed comparison of all methods with cost-effectiveness analysis
- **Concurrent Processing**: ThreadPoolExecutor for efficient API calls

## Requirements

- Python 3.10+
- Together AI API key
- Huggingface Token
- Required packages listed in `requirements.txt`
```

