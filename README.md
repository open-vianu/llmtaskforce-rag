# llmtaskforce-rag
Building a validated RAG system for Q&amp;A on regulatory documents  Background

# Installation

Install dependencies using poetry

```
if nvidia-smi &> /dev/null; then
  poetry install --extras "gpu"
else
  poetry install --extras "cpu"
fi
```
