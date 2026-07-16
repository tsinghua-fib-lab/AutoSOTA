# Invisible Character Preservation Tests

This repository reports a set of exploratory tests on the preservation of invisible Unicode characters across common web platforms and data sources that are known (or suspected) to be used for large-scale language model training.

## Experimental Setup

- **Browser:** Firefox (Linux)
- **Evaluation methods:**
  1. Inspecting the HTML source code of the webpage.
  2. Directly copying and pasting the rendered content from the webpage.

A sequence of **130 invisible Unicode characters** was inserted into each platform and tested for recoverability.

---

## Results

### LinkedIn

*(Link removed to preserve anonymity.)*

- **Copy–paste:** All 130 characters are successfully recovered.
- **HTML source code:** Only 25 characters are visible/recoverable. *(file removed to preserve anonymity.)*

This suggests that LinkedIn preserves most invisible characters at the rendering layer, but a significant fraction is lost or normalized at the HTML level.

---

### Wikipedia

*(Link removed to preserve anonymity.)*

- **Copy–paste:** All 130 characters are recovered.
- **HTML source code:** All 130 characters are preserved.

This indicates that Wikipedia fully preserves the inserted invisible characters.

---

### This README (Self-test)

The following string is embedded directly in this README file and can be used as a reference test:

A؜᠎​‌‍‎‏‪‬‭⁠⁡⁢⁣⁤⁦⁨⁩⁪⁫⁬⁭⁮⁯﻿𝅳𝅴𝅵𝅶𝅷𝅸𝅹𝅺󠀁󠀠󠀡󠀢󠀣󠀤󠀥󠀦󠀧󠀨󠀩󠀪󠀫󠀬󠀭󠀮󠀯󠀰󠀱󠀲󠀳󠀴󠀵󠀶󠀷󠀸󠀹󠀺󠀻󠀼󠀽󠀾󠀿󠁀󠁁󠁂󠁃󠁄󠁅󠁆󠁇󠁈󠁉󠁊󠁋󠁌󠁍󠁎󠁏󠁐󠁑󠁒󠁓󠁔󠁕󠁖󠁗󠁘󠁙󠁚󠁛󠁜󠁝󠁞󠁟󠁠󠁡󠁢󠁣󠁤󠁥󠁦󠁧󠁨󠁩󠁪󠁫󠁬󠁭󠁮󠁯󠁰󠁱󠁲󠁳󠁴󠁵󠁶󠁷󠁸󠁹󠁺󠁻󠁼󠁽󠁾󠁿B


All characters are preserved when copying this file directly.

---

### Reddit

Test URL:  
https://www.reddit.com/user/Chemical_Writer_8393/comments/1oobxn1/better_test/

- **Copy–paste:** All 130 characters are recovered.
- **HTML source code:** The characters are present but escaped, for example:

 "A\\u200b\\u200c\\u200d\\u2060\\ufeff\\ufe00\\ufe0f\\u034f\\u200e\\u200f\\u061cB"

 This shows that Reddit stores invisible characters in escaped Unicode form in the source code, while still allowing full recovery through copy–paste.

