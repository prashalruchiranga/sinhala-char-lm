from pathlib import Path
from datasets import load_dataset
import re
from tqdm import tqdm

class Cleaner():
    def __init__(self, data_export_dir: Path):
        self.cache_dir = data_export_dir.joinpath("cache")
        self.processed_dir = data_export_dir.joinpath("processed")
        self.processed_file_path = self.processed_dir.joinpath("cleaned.txt")

    @staticmethod
    def _create_1d_list(n_d_list: list[list[str]]) -> list[str]:
        one_d_list = []
        for row in n_d_list:
            if len(row) > 1:
                for sub_row in row:
                    one_d_list.append(sub_row)
            elif len(row) == 0:
                pass
            else:
                one_d_list.append(row[0])
        return one_d_list

    @staticmethod
    def _clean_text(text: str, replacement: str) -> str:
        print("Data cleaning in progress...")
        # lower english characters
        text = text.lower()
        # replace curly double quotes with straight double quotes
        text = re.sub(r"[\u201C\u201D]", "\u0022", text)
        # replace curly single quotes with straight single quotes
        text = re.sub(r"[\u2018\u2019]", "\u0027", text)
        # replace en-dash with dash
        text = re.sub(r"\u2013", "\u002D", text)
        # replace everything outside the allowed pattern with the replacement
        allowed_pattern = r"[a-z0-9\u0D80-\u0DFF\u200C\u200D!@#$%^&*()\[\]{}.,:;'\"<>?/\\|`~=_+ -]"
        # cleaned = ''.join(char if re.match(allowed_pattern, char) else replacement for char in text)
        cleaned_chars = []
        for char in tqdm(text, desc="Searching through text replacing OOV characters", unit="char"):
            if re.match(allowed_pattern, char):
                cleaned_chars.append(char)
            else:
                cleaned_chars.append(replacement)
        cleaned = ''.join(cleaned_chars)
        print("Data cleaning finished")
        return cleaned
    
    @staticmethod
    def _save(text: str, file_path: Path):
        directory = file_path.parent
        directory.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as file:
            file.write(text)
        print(f"Saved cleaned text to {file_path}")
        return None

    def __call__(self, hf_dataset:str, replacement: str = "", dataset_frac: float = 1.0):
        dataset = load_dataset(hf_dataset, cache_dir=self.cache_dir)
        raw_ds = dataset["train"]
        content = raw_ds["content"]
        text_list = Cleaner._create_1d_list(content)
        text = " ".join(text_list)
        if 0 < dataset_frac < 1.0:
            limit = int(dataset_frac * len(text))
            text = text[:limit]
        elif dataset_frac <= 0 or dataset_frac > 1.0:
            raise ValueError("dataset_frac must be a value between 0 and 1 (0, 1], includes 1.0")
        cleaned_text = Cleaner._clean_text(text, replacement)
        Cleaner._save(cleaned_text, self.processed_file_path)
        return self.processed_file_path
    