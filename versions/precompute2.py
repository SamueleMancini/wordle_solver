# precompute.py
import yaml
import numpy as np

def load_wordlist_txt(path):
    # loads a .txt with one word per line
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    words = list(dict.fromkeys(words))
    return words

def load_big_superset(path):
    """
    Loads a large list of 5-letter words from a YAML (or other) file. 
    The file might have thousands or tens of thousands of words.
    """
    data = yaml.load(open(path), Loader=yaml.FullLoader)
    # We'll assume 'data' is a list of strings.
    words = [w.lower().strip() for w in data if len(w)==5 and w.isalpha()]
    # remove duplicates, preserve order
    words = list(dict.fromkeys(words))
    return words

def compute_pattern(guess, answer):
    """
    Standard Wordle pattern => integer code 0..242 (3^5-1).
      2 => exact, 1 => misplaced, 0 => miss
    """
    pat = [0]*5
    ans_list = list(answer)

    # exact
    for i in range(5):
        if guess[i] == answer[i]:
            pat[i] = 2
            ans_list[i] = None

    # misplaced
    for i in range(5):
        if pat[i] == 0:
            ch = guess[i]
            if ch in ans_list:
                pat[i] = 1
                ans_list[ans_list.index(ch)] = None

    # convert base-3 => integer
    code = 0
    base = 1
    for d in pat:
        code += d*base
        base *= 3
    return code

if __name__=="__main__":
    # 1) Load superset from a YAML file, e.g. "big_superset.yaml"
    superset = load_wordlist_txt("guessable_words.txt")
    superset.sort()
    N = len(superset)
    print(f"Loaded {N} words in the superset.")

    # 2) Build NxN pattern matrix
    pattern_matrix = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        g = superset[i]
        for j in range(N):
            a = superset[j]
            pattern_matrix[i, j] = compute_pattern(g, a)

    # 3) Save
    np.savez("pattern_data.npz",
             words = np.array(superset, dtype=object),
             pattern_matrix = pattern_matrix)

    print("Finished building NxN matrix. Shape:", pattern_matrix.shape)
    print("Saved to pattern_data.npz")
