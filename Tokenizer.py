import numpy as np
import string
from dataloader import create_dataloader
from collections import Counter
from pathlib import Path

##  BPE tokenizer

DATA_FOLDER = Path(__file__).resolve().parent / "token_data"
VOCAB_PATH = DATA_FOLDER / "vocab.npy"
MERGE_PATH = DATA_FOLDER / "merge.npy"


class Tokenizer:

    def __init__(self, vocab_path: Path = VOCAB_PATH, merge_path: Path = MERGE_PATH, dataset_size: int = 100, vocab_size: int = 500) -> None:
        self.vocab: set[str] = set()
        self.merges: list[tuple[str, str]] = []
        self.ordered_vocab: np.ndarray = np.array([], dtype = str)
        self.token_to_id_lookup: dict[str, int] = {}
        self.id_to_token_lookup: dict[int, str] = {}

        self.vocab_path = vocab_path
        self.merge_path = merge_path

        self.dataset_size = dataset_size
        self.vocab_size = vocab_size

        self.dataloader = create_dataloader(batch_size= 1, limit=dataset_size)
        self.tokenized_stories: list[list[str]] = []


    def create_vocab(self) -> None:
        # build the first layer of single character tokens
        self.initialize_vocab()
        self.tokenize_dataset()

        # build the remaining vocabulary until we reach vocabulary_size
        self.run_tokenizer()
        # Save the current vocabulary to the local token_data folder.
        self.save_tokenizer()

    # adds the first layer of byte level tokens
    def initialize_vocab(self) -> None:
        self.vocab.update(string.ascii_lowercase)
        self.vocab.update(string.ascii_uppercase)
        self.vocab.update(
            {
                "<BOS>",
                "<EOS>",
                " ",
                ".",
                ",",
                "!",
                "?",
                "'",
                '"',
                ":",
                ";",
                "-",
                "(",
                ")",
                "[",
                "]",
                "\n",
                "…",
                "—",
                "–",
                "‘",
                "’",
                "“",
                "”",
            }
        )
        for batch in self.dataloader:
            self.vocab.update(set(batch[0]))

    # converts raw dataset strings into tokens "the" becomes "t", "h", "e"
    def tokenize_dataset(self) -> None:

        for batch in self.dataloader:
            story = batch[0]

            tokens = ["<BOS>"]
            tokens.extend(story) # cool python function that makes a list of chars out of string story
            tokens.append("<EOS>")
            self.tokenized_stories.append(tokens)



    def run_tokenizer(self) -> None:
        """

        while vocab.size < vocabulary_size do:
            initialize an empty counter for candidate tokens called candidates
            for i in enumerate(self.tokenized_stories):
                merge each pair of tokens
                dont perform this change in tokenized stories, only add this to candidate or increase counter
            add the highest candidate to vocab set
            record this merge step so that new text merges this token first (self.merge list[tuple[str, str]] for this)
                we do not want to perform longest matching as order of merges matters, first come tokens are more frequent
            for i in enumerate(self.tokenized_stories):
                compare the newly added token to this story
                if present add this new token and remove the merged ones
                scan LtoR, replace every nonoverlapping occurence of the new token

        """

        # we do not want to merge these ever
        special_tokens = {"<BOS>", "<EOS>"}

        while len(self.vocab) < self.vocab_size:
            # candidate stores each possible token merging possible in each round
            candidates: Counter[tuple[str, str]] = Counter()

            # Count adjacent pairs without modifying the stories.
            for story in self.tokenized_stories:
                for i in range(len(story) - 1):
                    pair = (story[i], story[i + 1])

                    if pair[0] in special_tokens or pair[1] in special_tokens:
                        continue

                    candidates[pair] += 1

            if not candidates:
                break

            best_pair, frequency = candidates.most_common(1)[0]

            # Avoid creating tokens that only occur once.
            if frequency < 2:
                break

            new_token = best_pair[0] + best_pair[1]

            # at this point we have a valid merging, record them
            self.vocab.add(new_token)
            self.merges.append(best_pair)

            # Replace the winning pair in every story.
            updated_stories: list[list[str]] = []

            for story in self.tokenized_stories:
                updated_story: list[str] = []
                i = 0

                while i < len(story):
                    if i < len(story) - 1 and story[i] == best_pair[0] and story[i + 1] == best_pair[1]:
                        updated_story.append(new_token)
                        i += 2 # skip both tokens
                    else:
                        updated_story.append(story[i])
                        i += 1

                updated_stories.append(updated_story)

            self.tokenized_stories = updated_stories

    def tokenize(self, text: str) -> list[str]:
        # after loading, we can use this function to tokenize new input

        # safety check added by codex
        if not self.merges or len(self.vocab) == 0:
            raise RuntimeError(
                "Tokenizer is not loaded. Call load_tokenizer() first."
            )

        # Start with character-level tokens.
        tokens = list(text)

        # Apply each learned merge in its original order.
        for left_token, right_token in self.merges:
            merged_token = left_token + right_token
            updated_tokens: list[str] = []

            index = 0

            while index < len(tokens):
                pair_matches = (
                    index + 1 < len(tokens)
                    and tokens[index] == left_token
                    and tokens[index + 1] == right_token
                )

                if pair_matches:
                    updated_tokens.append(merged_token)
                    index += 2
                else:
                    updated_tokens.append(tokens[index])
                    index += 1

            tokens = updated_tokens

        return ["<BOS>", *tokens, "<EOS>"]


    def tokens_to_ids(self, tokens: np.ndarray) -> np.ndarray:
        return np.array([self.token_to_id_lookup[str(token)] for token in tokens],dtype=np.int64)


    def ids_to_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        return self.ordered_vocab[token_ids]



    # we save numpy arrays of the sorted vocab and merge tuples
    # we also construct the lookup tables but do not save them
    def save_tokenizer(self) -> None:
        self.vocab_path.parent.mkdir(parents=True,exist_ok=True)
        self.merge_path.parent.mkdir(parents=True,exist_ok=True)

        # Sort to assign token IDs.
        # we build the vocab as a set for optimization but it is stored as np array to use index as token ids
        self.ordered_vocab = np.array(sorted(self.vocab),dtype=str)
        np.save(self.vocab_path, self.ordered_vocab)
        # the merges array provides the order of merging and is used to tokenize input text
        merges_array = np.array(self.merges,dtype=str)
        np.save(self.merge_path, merges_array)

        # Build the token-to-ID and ID-to-token lookup locally.
        self.token_to_id_lookup = {
            token: token_id
            for token_id, token in enumerate(self.ordered_vocab)
        }
        self.id_to_token_lookup = {
            token_id: str(token)
            for token_id, token in enumerate(self.ordered_vocab)
        }

        print(
            f"Saved {len(self.ordered_vocab):,} tokens "
            f"to {self.vocab_path}"
        )
        print(
            f"Saved {len(self.merges):,} merge rules "
            f"to {self.merge_path}"
        )
    

    # we load in self.ordered_vocab and self.merges
    # from there we setup self.vocab and the lookup tables
    # after this function, user can work with the tokenizer using a presaved generation
    def load_tokenizer(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and return the vocabulary and merge rules."""

        if not self.vocab_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.vocab_path}"
            )

        if not self.merge_path.is_file():
            raise FileNotFoundError(
                f"Merges file not found: {self.merge_path}"
            )

        self.ordered_vocab = np.load(self.vocab_path)

        merges_array = np.load(self.merge_path)

        # Restore the structures used by the tokenizer.
        self.vocab = set(self.ordered_vocab.tolist())

        self.merges = [
            (left, right)
            for left, right in merges_array.tolist()
        ]

        self.token_to_id_lookup = {
            token: token_id
            for token_id, token in enumerate(self.ordered_vocab)
        }

        self.id_to_token_lookup = {
            token_id: token
            for token_id, token in enumerate(self.ordered_vocab)
        }

        self.vocab_size = len(self.ordered_vocab)

        return self.ordered_vocab, merges_array



