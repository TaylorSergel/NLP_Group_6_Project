from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

language_codes = {
    "afrikaans": "afr",
    "isizulu": "zul",
    "english": "eng",
    "sesotho": "sot",
    "setswana": "tsn"
}

for lang_name, lang_code in language_codes.items():
    print(f"Downloading {lang_name} ({lang_code})...")
    try:
        ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", lang_code)
        for split in ds.keys():
            df = ds[split].to_pandas()
            df['language'] = lang_name
            df.to_csv(f"data/raw/brighter_{lang_name}_{split}.csv", index=False)
            print(f"  Saved data/raw/brighter_{lang_name}_{split}.csv ({len(df)} rows)")
    except Exception as e:
        print(f"  Could not load {lang_name}: {e}")

print("\nDone.")