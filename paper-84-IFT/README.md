# Time Series Forecasting
## Usage

1. Install Python 3.8. Then, install project dependencies with the following command.

   ```
   pip install -r requirements.txt
   ```

2. Prepare data. You can obtain raw datasets from this [[Link]](https://drive.google.com/file/d/1WHBqZxslQDjovTFxzDhzGDY5bKprdINJ/view?usp=sharing). Unzip and place the downloaded datasets in the root directory as `./datasets`.

3. Train and evaluate models with the commands provided in `./commands.txt`.

   ```
   bash scripts/ETT/IFT_ETTh1.sh
   ```

4. View forecasting results in the generated table `./table.txt`.

## Acknowledgement

This project is built based on the repository [Time-Series-Library](https://github.com/thuml/Time-Series-Library.git). We sincerely thank all corresponding contributors.
