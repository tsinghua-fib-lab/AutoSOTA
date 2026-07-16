# Robustness to Passive Transformations

This directory contains scripts, utilities, and experimental results for
evaluating the robustness of invisible Unicode characters under **passive,
real-world transformations**, including document format conversion, public APIs,
tokenizers, and web platforms.

These experiments aim to assess whether invisible watermarks are preserved under
common, non-adversarial text processing workflows.

---

## Key Takeaways

- All Unicode format and control characters were systematically evaluated for
  invisibility.
- A total of **130 characters** were selected for downstream experiments.
- Preservation strongly depends on the transformation:
  - PDF workflows are highly sensitive to fonts and generation methods.
  - Web platforms generally preserve invisible characters under copy–paste.
  - Tokenizers and APIs show heterogeneous behaviors.

---

## PDF Experiments

- **Python-based PDF generation and extraction**:  
  Up to 20 characters can be embedded and recovered.

- **Plain text → PDF → extraction**:  
  Only 2 characters are consistently preserved.

- **Manual copy–paste from PDF viewers**:  
  Preservation depends strongly on the viewer, with up to 20 characters
  preserved in the best case.

---

## Public APIs

Character preservation was evaluated using several public LLM APIs:

- **ChatGPT**: regurgitates 32 characters.
- **Le Chat**: preserves all 130 characters.
- **DeepSeek**: preserves all 130 characters from text, but none from PDF.

---

## Web Platforms

All 130 characters are preserved on common web platforms such as GitHub,
Wikipedia, and LinkedIn.

Depending on the platform, characters may appear:
- directly in rendered content, or
- in escaped form in the HTML source code.

---

## Scripts

### comparChar.py

Loads an input text file and compares invisible Unicode characters against a
reference text (`myText.txt`).  
Counts overlapping characters and outputs the results to
`common_characters.csv`.

---

## Files

### myText.txt

Reference text used for recognition and preservation experiments.  
Contains the 130 tested Unicode characters inserted between “A” and “B”.

---

## Experimental Modules

### Character selection

The `robustness/CharSelection/` directory documents the selection process for
invisible Unicode characters.

Candidate characters are collected from Unicode categories **Cc** (Control) and
**Cf** (Format), and filtered by visual inspection across multiple environments.

The final character set used in all experiments is:
- `robustness/CharSelection/alphabet.txt`

Main scripts:
- `getCharList.py` — Enumerates candidate characters.
- `embed.py` — Generates reference texts for invisibility testing.

---

### LLM API robustness

The `robustness/TestsAPI/` directory evaluates whether invisible watermarks are
preserved when texts are processed through different LLM APIs.

- `DeepSeek/` — DeepSeek API
- `LeChatAPI/` — Le Chat API
- `chatGPTAPI/` — ChatGPT API

For each API:
- `*FromTxt` — Direct text input.
- `*repetition` — Repeated interactions.

---

### PDF conversion robustness

The `robustness/toPdf/` directory evaluates preservation under PDF generation and
text extraction.

Experiments include:
- Direct PDF generation using Python.
- Python-based text extraction.
- Manual copy/paste from different PDF viewers.

Preservation strongly depends on the font and generation method, with direct
Python-based PDF generation providing the best overall results.

---

### Tokenizer robustness

The `robustness/tokenizer/` directory analyzes how invisible Unicode characters
are tokenized by different tokenizers.

- `tokenize_all.py` — Runs tokenization.
- `aggregate_tokenization.py` — Aggregates statistics.
- `plot_*.py` — Generates comparison figures.
- `*.csv` / `*.png` — Aggregated outputs.

---

### Web interface robustness

The `robustness/web/` directory evaluates whether invisible Unicode characters
are preserved under common web platforms through typical user actions
(copy/paste and HTML rendering).

- `gitCopyPaste/` — GitHub
- `linkedinCopyPaste/` — LinkedIn
- `redditCopyPaste/` — Reddit
- `wikiCopyPaste/` — Wikipedia
- `redditSrcCode/` — Reddit HTML source
- `wikiSrcCode/` — Wikipedia HTML source

## Repository structure

```
robustness/              # Robustness on non-adversarial transformations
├── CharSelection/
│   ├── alphabet.txt
│   ├── embed.py
│   ├── finalList.txt
│   ├── getCharList.py
│   ├── getCharList.txt
│   ├── invisible_unicode_characters.csv
│   └── readme.md
├── TestsAPI/
│   ├── DeepSeek/
│   │   ├── DSFromTxt
│   │   ├── DSrepeat
│   │   └── readme.md
│   ├── LeChatAPI/
│   │   ├── LeChatFromTxt
│   │   ├── LeChatrepetition
│   │   └── readme.md
│   ├── chatGPTAPI/
│   │   ├── ChatGPTfromTxt
│   │   ├── ChatGPTrepetition
│   │   └── readme.md
│   └── README.md
├── toPdf/
│   ├── create_pdf20.py
│   ├── test_create_pdf.py
│   ├── pdf2txt.py
│   ├── pdf2txt/
│   │   ├── CopyPasteAcrobat/
│   │   ├── CopyPasteChrome/
│   │   ├── CopyPasteEvince/
│   │   ├── CopyPasteFirefox/
│   │   ├── CopyPasteFirefoxWindows/
│   │   ├── CopyPasteGmail/
│   │   └── readme.md
│   ├── char_all.pdf
│   ├── char_all.txt
│   ├── char_select.pdf
│   ├── char_select.txt
│   ├── char_select_lo.pdf
│   ├── char_select_lo.txt
│   └── readme.md
├── tokenizer/
│   ├── Bert/
│   │   ├── test_bert.py
│   │   ├── bert_emoji.csv
│   │   └── bert_invisible_char.csv
│   ├── tokenize_all.py
│   ├── aggregate_tokenization.py
│   ├── plot_comparison.py
│   ├── plot_per_tokenizer.py
│   ├── plot_robust.py
│   ├── plot_robust_tokenizer.py
│   ├── aggregate_char.csv
│   ├── aggregate_tokenizer.csv
│   ├── invisible_char_token_counts_merged.csv
│   ├── emoji_tokenization_analysis_merged.csv
│   ├── emojisVsInvisTokens.png
│   ├── tokenizers.png
│   ├── robustTokenizers.png
│   ├── env.example
│   └── readme.md
├── web/
│   ├── gitCopyPaste/
│   ├── linkedinCopyPaste/
│   ├── redditCopyPaste/
│   ├── redditSrcCode/
│   ├── wikiCopyPaste/
│   ├── wikiSrcCode/
│   └── readme.md
├── comparChar.py        # Counts overlapping characters → common_characters.csv
├── myText.txt
└── readme.md
```
