# !curl --output palavras_pt_br.txt https://www.ime.usp.br/~pf/dicios/br-utf8.txt

import random
import string
from pathlib import Path

def read_words(file_path="./palavras_pt_br.txt"):
    """
        Reads the file with 261,788 portuguese words
        Remove duplicates 
    """
    
    with open(file_path, "r") as f:
        list_of_words = f.readlines()
        list_of_words = list(map(lambda x: x.strip("\n").lower(), list_of_words))

    list_of_words = sorted(list(set(list_of_words)), key=lambda x: len(x))
    
    return list_of_words


def random_key(n):
    key = ''.join(random.choices(string.ascii_uppercase, k=n))
    return key

def sample_words_and_keys(list_of_words, n, keys_size=4):
    """
        From the list of words, extract N random itens 
        Generate keys with fixed length to each word
    """
    
    sampled_words = random.sample(list_of_words, n)
    
    keys = []
    
    for i in range(n):
        # Generate a new random key that does not exist in `keys`
        while True:
            key = random_key(keys_size)
            if key not in keys:
                keys.append(key)
                break
            
    pairs = zip(keys, sampled_words)
    return pairs


def save_samples(pairs, n, output_directory="."):
    """Save key-word pairs to ``samples_K=<N>.txt`` and return its path."""

    output_path = Path(output_directory) / f"samples_K={n}.txt"

    with output_path.open("w", encoding="utf-8") as output_file:
        for key, word in pairs:
            output_file.write(f"{key} => {word}\n")

    return output_path


if __name__ == "__main__":
    for n in [32, 64, 128, 256, 512, 1024]:
        list_of_words = read_words()
        pairs = sample_words_and_keys(list_of_words, n=n, keys_size=4)
        output_path = save_samples(pairs, n)
        print(f"Saved {n} samples to {output_path}")
    
