# PaperPulse

PaperPulse is an exam paper pattern miner. It reads past exam papers (PDF, photo or text), maps each question to a chapter, and shows which topics are asked most, which questions repeat, and how well earlier papers predict later ones.

Live app: https://YOUR-LINK.streamlit.app

## Problem

Students revise without knowing which chapters carry the most marks or repeat most often. Past papers contain this information, but it is hard to see by reading them one by one.

## Features

- Upload PDFs, photos and text files. Scanned pages are read with OCR (Tesseract).
- Automatic question splitting and marks detection.
- Chapter mapping from an editable keyword syllabus, with presets for Nepal Grade 12 Mathematics and CBSE Class 12 Mathematics.
- Editable review table to correct chapters and marks.
- Revision priority chart, weighted so recent papers count more.
- Chapter by year heatmap of marks.
- Detection of repeated or paraphrased questions using TF-IDF cosine similarity.
- Backtest: predicts each paper's top chapters from earlier papers only and compares the result with random guessing.
- Topic search and CSV export.

## Method

1. Extract text from each file (pdfplumber, with OCR for scanned pages).
2. Split text into numbered questions and sum the marks written beside them.
3. Assign each question to the chapter with the most keyword matches.
4. Priority score = share of marks per chapter in each paper, averaged with higher weight on recent papers.
5. Backtest: for each paper from the third onward, predict its top chapters using only earlier papers, and count correct predictions.

## Results

Fill in after running on real papers:

- Papers analyzed: X (years ____ to ____)
- Questions analyzed: X
- Backtest: average X of 3 top chapters correct (random guessing: X of 3)
- Repeated question pairs found: X
- Students who used it: X

## Limitations

- Chapter mapping depends on the keyword list, so some questions may need manual correction.
- OCR can make mistakes on photos, so check the Review tab.
- With few papers the backtest is noisy. Patterns are not guarantees.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

For OCR on photos and scans, install Tesseract (on Ubuntu: `sudo apt install tesseract-ocr`).

## Tech

Python, Streamlit, pandas, scikit-learn, pdfplumber, pytesseract, Altair.
