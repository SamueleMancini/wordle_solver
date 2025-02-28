# guesser.py
import yaml
import numpy as np
from rich.console import Console

###############################################################################
# Load or define your curated guess set here
# If you prefer to load from "guessable_words.txt", do so in __init__ with fallback.
###############################################################################
CURATED_GUESS_SET = [
    "salet", "audio", "arise", "trace","roate","adieu","crane","slate","orate","soare","stare","irate",
    "light", "aeiou", "least", "slant", "pound","index","wryly","other","mound","harsh","clang","gamer","clint",
    "baste", "jaxey", "canoe", "tales", "frond","pluck","quick","sight","media","nomic","spore","grace","blink",
    "prick", "trial", "slice", "roast", "chime","smelt","ready","bride","piety","globe","knock","liver","tummy"
]
# This is just an example; you might have 200–300 words carefully chosen.

###############################################################################
# Feedback string to code
###############################################################################
def feedback_str_to_code(feedback_str):
    """
    Convert Wordle's feedback string (length=5) to an integer code (0..242).
      - '+' => 0 (miss)
      - '-' => 1 (misplaced)
      - else => 2 (exact)
    """
    code = 0
    base = 1
    for ch in feedback_str:
        if ch == '+':
            digit = 0
        elif ch == '-':
            digit = 1
        else:
            # If Wordle returns the actual letter, it means EXACT
            digit = 2
        code += digit * base
        base *= 3
    return code

###############################################################################
# Fallback pattern computation (for unknown words or guesses)
###############################################################################
def fallback_pattern(guess, answer):
    """
    Compute the Wordle pattern code (0..242) for guess vs. answer by:
      - Marking 'exact' matches (2) first,
      - Then marking 'misplaced' (1),
      - Else 'miss' (0).
    """
    pat = [0]*5
    ans_list = list(answer)  # mutable copy

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

###############################################################################
# Shannon entropy
###############################################################################
def shannon_entropy(counts):
    total = counts.sum()
    if total <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return -np.sum(p * np.log2(p))

###############################################################################
# Guesser class
###############################################################################
class Guesser:
    def __init__(self, manual):
        """
        1) Load the current word list from 'wordlist.yaml'.
        2) Load the precomputed data from 'pattern_data.npz'.
        3) Split words into known vs. unknown.
        4) Build a curated small-guess array from CURATED_GUESS_SET.
        5) Set thresholds for two-phase approach.
        6) Prepare to do partial lookahead for small sets.
        """
        self.console = Console()
        self._manual = manual

        # 1) Load the current word list (solutions) - might be train/dev/test
        self.word_list = yaml.load(open('wordlist.yaml'), Loader=yaml.FullLoader)
        # Ensure uniqueness
        self.word_list = list(dict.fromkeys(self.word_list))

        # 2) Load precomputed data
        data = np.load('pattern_data.npz', allow_pickle=True)
        self.precomp_words = list(data['words'])         # shape (N,)
        self.pattern_matrix = data['pattern_matrix']     # shape (N, N)

        # Build map from word -> index
        self.precomp_idx = {w: i for i, w in enumerate(self.precomp_words)}

        # 3) Split into known vs. unknown
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

        # 4) Build a curated small-guess array
        #    from the intersection of CURATED_GUESS_SET & precomp dictionary
        curated_indices = []
        for gw in CURATED_GUESS_SET:
            idx = self.precomp_idx.get(gw, -1)
            if idx >= 0:
                curated_indices.append(idx)
        # If empty, fallback to first 300 words in precomp
        if len(curated_indices) == 0:
            curated_indices = list(range(min(len(self.precomp_words), 300)))

        self.curated_guess_idxs = np.array(curated_indices, dtype=np.int32)

        # 5) Set thresholds
        # Large -> guess from curated set. Then switch to full candidate set.
        self.SWITCH_THRESHOLD = 350

        # 6) Depth-2 lookahead threshold for small sets
        self.LOOKAHEAD_THRESHOLD = 15


        # Keep track of guesses & feedback
        self._guesses = []
        self._feedbacks = []

    def restart_game(self):
        # Reset at start of each new word
        self.candidate_known = self.known_idxs.copy()
        self.candidate_unknown = list(self.unknown_words)
        self._guesses.clear()
        self._feedbacks.clear()

    def filter_candidates(self, guess, pattern_code):
        """
        Eliminate candidates (both known & unknown) that would not yield 'pattern_code'
        if 'guess' was the solution.
        """
        guess_idx = self.precomp_idx.get(guess, -1)

        # Filter known
        if guess_idx >= 0:
            row = self.pattern_matrix[guess_idx, self.candidate_known]
            mask = (row == pattern_code)
            self.candidate_known = self.candidate_known[mask]
        else:
            # fallback
            filtered = []
            for ans_idx in self.candidate_known:
                ans_word = self.precomp_words[ans_idx]
                code = fallback_pattern(guess, ans_word)
                if code == pattern_code:
                    filtered.append(ans_idx)
            self.candidate_known = np.array(filtered, dtype=np.int32)

        # Filter unknown
        new_unk = []
        for uw in self.candidate_unknown:
            code = fallback_pattern(guess, uw)
            if code == pattern_code:
                new_unk.append(uw)
        self.candidate_unknown = new_unk

    def compute_distribution(self, guess):
        """
        Return a frequency array freq[pattern] for how many candidate solutions
        would produce 'pattern' (0..242) if we guess 'guess'.
        """
        freq = np.zeros(243, dtype=int)
        gidx = self.precomp_idx.get(guess, -1)

        # Known
        if gidx >= 0:
            row = self.pattern_matrix[gidx, self.candidate_known]
            for code in row:
                freq[code] += 1
        else:
            # fallback
            for ans_idx in self.candidate_known:
                ans_word = self.precomp_words[ans_idx]
                code = fallback_pattern(guess, ans_word)
                freq[code] += 1

        # Unknown
        for uw in self.candidate_unknown:
            code = fallback_pattern(guess, uw)
            freq[code] += 1

        return freq

    ############################################################################
    # Depth-2 lookahead for small sets
    ############################################################################
    def depth2_score(self, guess):
        """
        Returns an approximate "expected number of candidates after 2 guesses"
        or something akin to a 2-step average size. This is not an exact solver,
        but it often helps reduce the final guess count if the set is very small.
        """
        # distribution over patterns
        freq = self.compute_distribution(guess)
        total = freq.sum()
        if total <= 1:
            return 0.0

        # Weighted sum of next-step sizes
        # For each pattern p, we approximate how large the subset remains
        # if we guess 'guess' and get pattern p. Then we take the weighted average.
        # That is smaller => better.

        sum_next = 0.0

        # We'll need to filter to see how many remain for each pattern p
        # But we do a quick approach: for known solutions, we can do row lookups again.

        guess_idx = self.precomp_idx.get(guess, -1)

        # We'll build subsets for known & unknown, for each pattern
        # This can be somewhat costly, but it's only done if set < LOOKAHEAD_THRESHOLD
        # => so it's small enough.

        # First gather them into pattern -> list of candidates
        pattern_buckets_known = {}
        pattern_buckets_unknown = {}

        # Build known pattern partition
        if guess_idx >= 0:
            # we can do row lookups
            row = self.pattern_matrix[guess_idx, self.candidate_known]
            for cand_idx, pat_code in zip(self.candidate_known, row):
                pattern_buckets_known.setdefault(pat_code, []).append(cand_idx)
        else:
            # fallback
            for cand_idx in self.candidate_known:
                ans_word = self.precomp_words[cand_idx]
                pat_code = fallback_pattern(guess, ans_word)
                pattern_buckets_known.setdefault(pat_code, []).append(cand_idx)

        # Build unknown pattern partition
        for uw in self.candidate_unknown:
            pat_code = fallback_pattern(guess, uw)
            pattern_buckets_unknown.setdefault(pat_code, []).append(uw)

        # Now for each pattern p, we see how big that subset is
        for pcode, count in enumerate(freq):
            if count == 0:
                continue
            # the subset size is count
            # for a rough next-step measure, we do not recalc the best next guess,
            # but let's estimate an "average" or "representative" measure

            # We can do a smaller sub-entropy approach, but let's do a simpler measure: subset size
            # e.g. next_size = # of known + # of unknown in that pattern bucket
            known_bucket = pattern_buckets_known.get(pcode, [])
            unknown_bucket = pattern_buckets_unknown.get(pcode, [])
            next_size = len(known_bucket) + len(unknown_bucket)

            sum_next += next_size * (count / total)

        return sum_next

    ############################################################################
    # pick_guess_smallset / pick_guess_full / pick_guess_depth2
    ############################################################################
    def pick_guess_smallset(self):
        """
        Use curated guess set to pick guess. We do single-step entropy
        to select the guess with the highest info gain.
        """
        best_score = -1.0
        best_guess = None
        for idx in self.curated_guess_idxs:
            guess_word = self.precomp_words[idx]
            freq = self.compute_distribution(guess_word)
            score = shannon_entropy(freq)
            if score > best_score:
                best_score = score
                best_guess = guess_word
        if best_guess is None:
            return "salet"
        return best_guess

    def pick_guess_full(self):
        """
        Pick guess from the entire candidate set (known) by single-step entropy
        or fallback if no known remain -> guess among unknown.
        """
        # We do single-step entropy among candidate_known
        best_score = -1.0
        best_guess = None

        for idx in self.candidate_known:
            guess_word = self.precomp_words[idx]
            freq = self.compute_distribution(guess_word)
            score = shannon_entropy(freq)
            if score > best_score:
                best_score = score
                best_guess = guess_word

        # If no known remain
        if best_guess is None and len(self.candidate_unknown) > 0:
            for uw in self.candidate_unknown:
                freq = self.compute_distribution(uw)
                score = shannon_entropy(freq)
                if score > best_score:
                    best_score = score
                    best_guess = uw

        if best_guess is None:
            return "salet"
        return best_guess

    def pick_guess_depth2(self):
        """
        If the candidate set is very small, we can do a partial depth-2 lookahead
        to pick the guess with minimal expected subspace size.
        This is an alternative to single-step entropy. 
        """
        # We'll guess from candidate_known, or if none remain, from candidate_unknown
        # Then pick the guess that yields the smallest expected next subset size.
        best_val = 1e9
        best_guess = None

        # Try known guesses
        for idx in self.candidate_known:
            guess_word = self.precomp_words[idx]
            val = self.depth2_score(guess_word)  # smaller is better
            if val < best_val:
                best_val = val
                best_guess = guess_word

        # If no known remain, check unknown
        if best_guess is None and len(self.candidate_unknown) > 0:
            for uw in self.candidate_unknown:
                val = self.depth2_score(uw)
                if val < best_val:
                    best_val = val
                    best_guess = uw

        if best_guess is None:
            return "salet"
        return best_guess

    ############################################################################
    # get_guess
    ############################################################################
    def get_guess(self, last_feedback):
        # If manual mode, get input from console
        if self._manual == 'manual':
            return self.console.input("[bold magenta]Your guess[/bold magenta]: ")

        # First guess?
        if not self._guesses:
            # use a stable known opener
            opener = "salet"
            if opener not in self.precomp_idx:
                # fallback
                if len(self.candidate_known) > 0:
                    opener = self.precomp_words[self.candidate_known[0]]
                else:
                    opener = "crane"
            self._guesses.append(opener)
            return opener

        # Convert feedback to pattern code
        pattern_code = feedback_str_to_code(last_feedback)

        # Filter candidate sets
        last_guess = self._guesses[-1]
        self.filter_candidates(last_guess, pattern_code)

        # If exactly 1 candidate remains
        total_cand = len(self.candidate_known) + len(self.candidate_unknown)
        if total_cand == 1:
            if len(self.candidate_known) == 1:
                final_guess = self.precomp_words[self.candidate_known[0]]
            else:
                final_guess = self.candidate_unknown[0]
            self._guesses.append(final_guess)
            return final_guess

        # Otherwise, pick next guess
        if total_cand >= self.SWITCH_THRESHOLD:
            # Large set => guess from curated small set
            next_guess = self.pick_guess_smallset()
        else:
            # If set is not too big, we can guess from full set
            # but if it's REALLY small (< LOOKAHEAD_THRESHOLD), do partial lookahead
            if total_cand < self.LOOKAHEAD_THRESHOLD:
                next_guess = self.pick_guess_depth2()
            else:
                next_guess = self.pick_guess_full()

        self._guesses.append(next_guess)
        return next_guess
