## Baseline and Code Modification

This project uses the official **datawatermarks** repository as the baseline
implementation:

https://github.com/ryanyxw/datawatermarks/tree/main

Since the data format used in our experiments differs from the original baseline,
we introduce a modified script **`perturb_modified.py`** to replace the original
**`perturb.py`** for compatibility.

> **Important:** The only difference between `perturb_modified.py` and the
> original `perturb.py` lies in the implementation of the function
> `edit_json_unicode()`. All other functionalities remain identical to the
> baseline.

## Usage

1. **Clone the baseline repository**

```bash
git clone https://github.com/ryanyxw/datawatermarks.git
```

2. **Copy the modified script into the source directory**

    Place `perturb_modified.py` into the following path:
    ```bash
    datawatermarks/src/perturb_modified.py
    ```
3. **Run experiments as usual**

All subsequent baseline comparison perturbation steps will use `perturb_modified.py` instead of the original `perturb.py`, without requiring any other code changes.
