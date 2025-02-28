import yaml
import numpy as np
from rich.console import Console


GUESS_LIST = [
    "salet","trace","roate","adieu","crane","slate","orate","soare","stare","irate",
    "light","pound","index","wryly","other","mound","harsh","clang","gamer","clint",
    "baste","frond","pluck","quick","sight"
]


# Fallback pattern computation for unknown words or unknown guesses
def compute_pattern(guess, answer):
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

    # Convert base-3 pattern to integer
    code = 0
    base = 1
    for digit in pat:
        code += digit * base
        base *= 3
    return code


# Convert Wordle's feedback string to an integer code (0..242)
def feedback_str_to_code(feedback_str):
    code = 0
    base = 1
    for ch in feedback_str:
        if ch == '+':
            digit = 0
        elif ch == '-':
            digit = 1
        else:
            digit = 2
        code += digit * base
        base *= 3
    return code


# Entropy calculation
def calc_entropy(counts):
    total = counts.sum()
    if total <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))


class Guesser:
    def __init__(self, manual):
        self.console = Console()
        self._manual = manual

        self.word_list = yaml.load(open('wordlist.yaml'), Loader=yaml.FullLoader)

        # Load precomputed data
        data = np.load('pattern_data.npz', allow_pickle=True)
        self.precomp_words = list(data["words"])         
        self.pattern_matrix = data["pattern_matrix"]       

        # Make a map word -> index in the precomputed array
        self.precomp_idx = {w: i for i, w in enumerate(self.precomp_words)}

        # Split word_list into known vs. unknown
        known_list = []
        unknown_list = []
        for w in self.word_list:
            idx = self.precomp_idx.get(w, -1)
            if idx == -1:
                unknown_list.append(w)
            else:
                known_list.append(idx)

        self.known_idxs = np.array(known_list, dtype=np.int32)
        self.unknown_words = unknown_list

        self.candidate_known = self.known_idxs.copy() 
        self.candidate_unknown = list(self.unknown_words)

        # Build a smaller guess set from GUESS_LIST
        self.guess_candidates = []
        for guess_word in GUESS_LIST:
            if guess_word in self.precomp_idx:
                self.guess_candidates.append(self.precomp_idx[guess_word])
        self.guess_candidates = np.array(self.guess_candidates, dtype=np.int32)

        if len(self.guess_candidates) == 0:
            self.guess_candidates = np.arange(min(300, len(self.precomp_words)), dtype=np.int32)

        # Tuning parameter: threshold to switch from the small guess set to the actual candidate set
        self.SWITCH_THRESHOLD = 350

        self._tried = []
        self._feedbacks = []


    def restart_game(self):
        self.candidate_known = self.known_idxs.copy()
        self.candidate_unknown = list(self.unknown_words)
        self._tried = []
        self._feedbacks = []


    # Filter candidate sets (both known & unknown)
    def filter_candidates(self, guess, pattern_code):

        guess_idx = self.precomp_idx.get(guess, -1)

        # Filter known candidates
        if guess_idx >= 0:
            row = self.pattern_matrix[guess_idx, self.candidate_known]
            mask = (row == pattern_code)
            self.candidate_known = self.candidate_known[mask]
        else:
            filtered = []
            for ans_idx in self.candidate_known:
                ans_word = self.precomp_words[ans_idx]
                code = compute_pattern(guess, ans_word)
                if code == pattern_code:
                    filtered.append(ans_idx)
            self.candidate_known = np.array(filtered, dtype=np.int32)

        # Filter unknown candidates
        new_unknown = []
        for ans_word in self.candidate_unknown:
            code = compute_pattern(guess, ans_word)
            if code == pattern_code:
                new_unknown.append(ans_word)
        self.candidate_unknown = new_unknown


    # Compute frequency array
    def compute_distribution(self, guess):
       
        freq = np.zeros(243, dtype=int)
        guess_idx = self.precomp_idx.get(guess, -1)

        # Known solutions
        if guess_idx >= 0:
            row = self.pattern_matrix[guess_idx, self.candidate_known]
            for code in row:
                freq[code] += 1
        else:
            for ans_idx in self.candidate_known:
                ans_word = self.precomp_words[ans_idx]
                code = compute_pattern(guess, ans_word)
                freq[code] += 1

        # Unknown solutions
        for ans_word in self.candidate_unknown:
            code = compute_pattern(guess, ans_word)
            freq[code] += 1

        return freq


    # Pick next guess maximizing the entropy among self.guess_candidates (used when candidate set >= SWITCH_THRESHOLD)
    def pick_guess_smallset(self):

        best_guess = None
        best_score = -1.0

        for idx in self.guess_candidates:
            guess_word = self.precomp_words[idx]
            freq = self.compute_distribution(guess_word)
            score = calc_entropy(freq)
            if score > best_score:
                best_score = score
                best_guess = guess_word

        if best_guess is None:
            return "salet"
        return best_guess


    # Pick next guess maximizing the entropy among all current candidate words (used when candidate set < SWITCH_THRESHOLD)
    def pick_guess_full(self):

        best_guess = None
        best_score = -1.0

        for idx in self.candidate_known:
            guess_word = self.precomp_words[idx]
            freq = self.compute_distribution(guess_word)
            score = calc_entropy(freq)
            if score > best_score:
                best_score = score
                best_guess = guess_word

        # If we run out of known candidates, guess among unknown
        if best_guess is None and len(self.candidate_unknown) > 0:
            for guess_word in self.candidate_unknown:
                freq = self.compute_distribution(guess_word)
                score = calc_entropy(freq)
                if score > best_score:
                    best_score = score
                    best_guess = guess_word

        if best_guess is None:
            return "salet"
        return best_guess


    def get_guess(self, last_feedback):
        
        if self._manual == 'manual':
            return self.console.input("Your guess: ")

        # First guess
        if not self._tried:
            # Choose a known opener 
            opener = "salet"
            if opener not in self.precomp_idx:
                if len(self.candidate_known) > 0:
                    opener_idx = self.candidate_known[0]
                    opener = self.precomp_words[opener_idx]
                else:
                    opener = "crane"
            self._tried.append(opener)
            return opener

        # Convert last_feedback to pattern code
        pattern_code = feedback_str_to_code(last_feedback)

        # Filter candidates
        last_guess = self._tried[-1]
        self.filter_candidates(last_guess, pattern_code)

        # If only 1 candidate left, guess it
        total_candidates = len(self.candidate_known) + len(self.candidate_unknown)
        if total_candidates == 1:
            if len(self.candidate_known) == 1:
                final_guess = self.precomp_words[self.candidate_known[0]]
            else:
                final_guess = self.candidate_unknown[0]
            self._tried.append(final_guess)
            return final_guess

        # Otherwise pick the next guess
        if total_candidates >= self.SWITCH_THRESHOLD:
            # Use the smaller guess set for speed
            next_guess = self.pick_guess_smallset()
        else:
            # Switch to guessing from the full candidate set
            next_guess = self.pick_guess_full()

        self._tried.append(next_guess)
        return next_guess
