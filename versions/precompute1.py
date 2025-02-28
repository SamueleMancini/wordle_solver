# precompute.py
import yaml
import numpy as np

###############################################################################
# 1) Load / merge multiple word lists
###############################################################################
def load_wordlist_yaml(path):
    # loads a .yaml list of words (the format we typically have)
    return yaml.load(open(path), Loader=yaml.FullLoader)

def load_wordlist_txt(path):
    # loads a .txt with one word per line
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                out.append(w)
    return out

# Add or remove paths as desired
train_words = load_wordlist_yaml('wordlist.yaml')  # or 'wordlist.yaml' if you want
dev_words   = load_wordlist_yaml('dev_wordlist.yaml')
# test_words  = load_wordlist_yaml('test_wordlist.yaml')  # if you have it available
guess_list  = load_wordlist_txt('guessable_words.txt')    # optional curated guess set

# Merge them
all_words = set()
for lst in [train_words, dev_words, guess_list]:
    if lst:
        for w in lst:
            all_words.add(w.lower().strip())

# Convert to a sorted list
merged_words = sorted(all_words)

###############################################################################
# 2) Define a pattern function (Wordle feedback => integer code)
###############################################################################
def compute_pattern(guess, answer):
    """
    Return an integer code 0..242 representing the Wordle feedback for guess vs. answer.
    5-letter words, digits in base-3:
      2 => exact, 1 => misplaced, 0 => miss
    """
    pat = [0]*5
    ans_list = list(answer)

    # Mark exact
    for i in range(5):
        if guess[i] == answer[i]:
            pat[i] = 2
            ans_list[i] = None

    # Mark misplaced
    for i in range(5):
        if pat[i] == 0:
            ch = guess[i]
            if ch in ans_list:
                pat[i] = 1
                ans_list[ans_list.index(ch)] = None

    # Convert from base-3 to integer
    code = 0
    base = 1
    for digit in pat:
        code += digit * base
        base *= 3
    return code

###############################################################################
# 3) Build the pattern matrix (N x N)
###############################################################################
N = len(merged_words)
print(f"Building pattern matrix for {N} words...")

pattern_matrix = np.zeros((N, N), dtype=np.uint8)
for i in range(N):
    guess_word = merged_words[i]
    for j in range(N):
        answer_word = merged_words[j]
        pattern_matrix[i, j] = compute_pattern(guess_word, answer_word)

###############################################################################
# 4) Save to pattern_data.npz
###############################################################################
np.savez('pattern_data.npz',
         words = np.array(merged_words, dtype=object),  # store as object/string
         pattern_matrix = pattern_matrix)

print("Done. Created pattern_data.npz with:", len(merged_words), "words.")
