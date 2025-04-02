# llmtaskforce-rag
Building a validated RAG system for Q&amp;A on regulatory documents  Background

# Scripts

`plot_results.py`: Plot the results found in the Markdown file at `eval/RESULTS.md` and places a barplot in `eval/RESULTS.png`

# Installation

Install dependencies using poetry

```
if nvidia-smi &> /dev/null; then
  poetry install --extras "gpu"
else
  poetry install --extras "cpu"
fi
```
