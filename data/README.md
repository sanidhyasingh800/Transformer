# Local datasets

`tinystories_gpt4_clean.parquet` is downloaded from
[`karpathy/tinystories-gpt4-clean`](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).

The Parquet file contains one `text` column and 2,732,634 stories. The dataset
publisher suggests using the pre-shuffled rows as follows:

- test: rows 0 through 9,999
- validation: rows 10,000 through 19,999
- train: rows 20,000 through the end

The downloaded Parquet file is intentionally ignored by Git.
