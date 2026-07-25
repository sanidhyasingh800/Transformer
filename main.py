import numpy as np
from tokenizer import Tokenizer

tokenizer = Tokenizer(dataset_size=5000, vocab_size=100)


loaded_vocab = tokenizer.load_tokenizer()

print(tokenizer.tokenize("The Stack is a collection of source code from repositories with various licenses. Any use of all or part of the code gathered in The Stack must abide by the terms of the original licenses, including attribution clauses when relevant. We facilitate this by providing provenance information for each data point."))