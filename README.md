# MAKER: Agentic Tower of Hanoi Solver (SOLVING A MILLION-STEP LLM TASK WITH ZERO ERRORS)

This project implements the **MAKER** algorithm (as described in recent research) to solve the Tower of Hanoi puzzle using an LLM-based agent. It employs a "First-to-ahead-by-k" voting mechanism and a Red-Flagging parser to ensure robustness and accuracy, even with smaller local models.

# Paper
https://arxiv.org/pdf/2511.09030

## Features

- **Agentic Solver**: Uses an LLM (Large Language Model) to decide the next move.
- **Robust Voting Mechanism**: Implements "First-to-ahead-by-k" voting to filter out noise and hallucinations.
- **Red-Flagging**: Automatically discards invalid formats and illegal moves (hallucinations).
- **Error Rate Tracking**: Real-time calculation of the model's error rate and system accuracy.
- **Local LLM Support**: Designed to work with local models via LM Studio (or any OpenAI-compatible API).

## Prerequisites

1.  **Python 3.8+**
2.  **LM Studio** (or another local inference server running on port 1234)
    *   *Note: You can configure the `base_url` in `maker.py` if your server runs elsewhere.*
3.  **A Local Model**: We recommend models like `phi-3`, `mistral`, or `llama-3`.

## Installation

1.  Clone this repository or download the files.
2.  Install the required Python package:

    ```bash
    pip install openai
    ```

## Configuration

Open `maker.py` to adjust settings:

```python
# Connect to LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Parameters
K_THRESHOLD = 10   # Voting confidence threshold (higher = safer but slower)
NUM_DISKS = 3      # Number of disks for the puzzle
```

## Usage

1.  Start your Local LLM server (e.g., LM Studio).
2.  Run the script:

    ```bash
    python maker.py
    ```

## Output Explanation

The script prints the progress of each step:
- **Votes**: The distribution of moves suggested by the model.
- **Margin**: The lead of the top move over the runner-up.
- **Cumulative Error Rate**: The percentage of invalid/illegal responses from the model so far.

At the end, it displays the **System Accuracy**, which represents the percentage of time the model produced a valid, legal move.

## License

MIT

## Local results 

Loop until solved mode

NUM_DISKS = 3  
K_THRESHOLD = 10
MAX_TOKENS = 750 
Model OpenAI/gpt-oss-20b
temperature=0.7
Context Limit on model is set to 2000

1s run
SOLVED in 14 steps!
Final Statistics:
  Total Samples: 199
  Total Errors:  61
  Final Error Rate: 30.7%
  System Accuracy:  69.3%

2nd run
SOLVED in 14 steps!
Final Statistics:
  Total Samples: 198
  Total Errors:  58
  Final Error Rate: 29.3%
  System Accuracy:  70.7%