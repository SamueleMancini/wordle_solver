# precompute.py
import yaml
import numpy as np

# ---------------------------
# 1) Load your 5-letter words
# ---------------------------
word_list = yaml.load(open('guessable.yaml'), Loader=yaml.FullLoader)
word_list = sorted(set(word_list))  # Sort & ensure unique

# Build a mapping from word -> index
word2idx = {w: i for i, w in enumerate(word_list)}
n = len(word_list)

# ---------------------------
# 2) Define a pattern function
# ---------------------------
def compute_pattern(guess, answer):
    """
    Computes the Wordle feedback pattern for guess vs. answer as an integer 0..242
    using the standard 0=miss, 1=misplaced, 2=exact scheme in base 3.
    """
    # 'pattern' array of length 5
    pattern = [0]*5

    # Mark exact matches
    answer_chars = list(answer)
    for i in range(5):
        if guess[i] == answer[i]:
            pattern[i] = 2
            answer_chars[i] = None

    # Mark misplaced matches
    for i in range(5):
        if pattern[i] == 0:
            ch = guess[i]
            if ch in answer_chars:
                pattern[i] = 1
                # remove the matched char so we don't reuse it
                answer_chars[answer_chars.index(ch)] = None

    # Convert base-3 array to integer
    code = 0
    multiple = 1
    for digit in pattern:
        code += digit * multiple
        multiple *= 3
    return code

# --------------------------------------------
# 3) Build the (n x n) matrix of pattern codes
# --------------------------------------------
# pattern_matrix[i, j] = integer code for guess=word_list[i], answer=word_list[j]
pattern_matrix = np.zeros((n, n), dtype=np.uint8)

for i in range(n):
    g = word_list[i]
    for j in range(n):
        a = word_list[j]
        pattern_matrix[i, j] = compute_pattern(g, a)

# -------------------------------------
# 4) Save to an .npz file for quick load
# -------------------------------------
np.savez('pattern_data.npz',
         words = word_list,
         pattern_matrix = pattern_matrix)
print("Precomputation finished. pattern_data.npz created.")
