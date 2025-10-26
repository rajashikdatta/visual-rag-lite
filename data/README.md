# Data Directory

This directory contains datasets for training and evaluation.

## Structure

```
data/
├── docvqa/
│   ├── train/
│   ├── val/
│   └── test/
├── infographicvqa/
│   ├── train/
│   ├── val/
│   └── test/
└── raw/
```

## Datasets

### DocVQA

Download from: https://rrc.cvc.uab.es/?ch=17

The DocVQA dataset contains:
- 12,767 document images
- 50,000 questions

Place the downloaded files in the `docvqa/` directory following the structure above.

### InfographicVQA

Download from: https://www.docvqa.org/datasets/infographicvqa

The InfographicVQA dataset contains:
- 5,485 infographics
- 30,035 questions

Place the downloaded files in the `infographicvqa/` directory.

## Data Format

Each dataset should have annotation files in JSON format:
```json
{
  "data": [
    {
      "question_id": 1,
      "question": "What is the total?",
      "answers": ["$1000", "1000 dollars"],
      "document_id": "doc_001",
      "image_path": "images/doc_001.png"
    }
  ]
}
```
