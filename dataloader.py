"""Simple TinyStories DataLoader."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "tinystories_gpt4_clean.parquet"
)

NUM_STORIES = 20_000

# TinyStories Dataset model, returns the first NUM_STORIES stories in the dataset
# used by d
class TinyStoriesDataset(Dataset):
    def __init__(self, dataset_path: Path = DATASET_PATH,limit: int = NUM_STORIES) -> None:
        table = pq.read_table(dataset_path,columns=["text"])
        self.stories = table["text"].to_pylist()[:limit]

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, index: int) -> str:
        return self.stories[index]


def create_dataloader(batch_size: int = 64, limit: int = NUM_STORIES) -> DataLoader:
    dataset = TinyStoriesDataset(limit = limit)

    return DataLoader(dataset, batch_size=batch_size,shuffle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--printStories",
        action="store_true",
        help="print sample stories and dataset statistics",
    )
    args = parser.parse_args()

    if not args.printStories:
        return

    dataloader = create_dataloader(batch_size=64)
    dataset = TinyStoriesDataset()
    for i, batch in enumerate(dataloader):
        if i == 0:
            print(f"Batch size: {len(batch)}")
        if i % 100 == 0:
            print(batch[0])

    print(f"Total stories: {len(dataset):,}")


if __name__ == "__main__":
    main()
