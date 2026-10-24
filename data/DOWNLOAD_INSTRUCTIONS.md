# FracAtlas dataset

This project trains on FracAtlas. The dataset is not committed to this repository because the archive is large and has separate distribution terms.

1. Obtain FracAtlas from its official distribution page and review its license/terms before downloading or redistributing it.
2. Extract the archive so the directory contains:

   ```text
   FracAtlas/
     dataset.csv
     images/
       Fractured/
       Non_fractured/
   ```

3. Train from the repository root:

   ```powershell
   python main.py --dataset fracatlas --data_dir "path\to\FracAtlas" --epochs 15 --workers 0
   ```

`--workers 0` avoids Windows multiprocessing issues. FracAtlas does not provide patient IDs, so the project uses a random image-level split and its test metrics may be optimistic.
