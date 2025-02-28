# guesser.py
import yaml
import numpy as np
from rich.console import Console

###############################################################################
# fallback pattern for unknown guess/solution pairs
###############################################################################
def fallback_pattern(guess, answer):
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
    # convert base-3 => int
    code=0
    base=1
    for d in pat:
        code += d*base
        base*=3
    return code

###############################################################################
# Convert Wordle feedback string => integer pattern code
###############################################################################
def feedback_str_to_code(feedback_str):
    code=0
    base=1
    for ch in feedback_str:
        if ch=='+':
            digit=0
        elif ch=='-':
            digit=1
        else:
            digit=2
        code += digit*base
        base*=3
    return code

###############################################################################
# Shannon entropy
###############################################################################
def shannon_entropy(freq):
    total = freq.sum()
    if total<=1:
        return 0.0
    p = freq[freq>0]/ total
    return -np.sum(p* np.log2(p))

###############################################################################
class Guesser:
    def __init__(self, manual):
        self.console = Console()
        self._manual = manual
        self._guesses = []
        self._data_loaded = False
        self.load_data()
    
    def load_data(self):
        """Loads the wordlist and pattern data once, initializing the necessary attributes."""
        if self._data_loaded:
            return
        
        # Load test set from wordlist.yaml
        test_set = yaml.load(open("dev_wordlist.yaml"), Loader=yaml.FullLoader)
        test_set = list(dict.fromkeys(test_set))  # unique
        test_set = [w.lower().strip() for w in test_set]
        
        # Load big NxN superset
        data = np.load("pattern_data.npz", allow_pickle=True)
        self.superset_words = list(data["words"])
        self.pattern_matrix_full = data["pattern_matrix"]
        
        # Build index mappings
        self.superset_idx = {w: i for i, w in enumerate(self.superset_words)}
        
        # Identify test words in superset
        self.in_superset = [w for w in test_set if w in self.superset_idx]
        self.not_in_superset = [w for w in test_set if w not in self.superset_idx]
        
        self.col_indices = np.array([self.superset_idx[w] for w in self.in_superset], dtype=np.int32)
        self.test_words_in_matrix = self.in_superset
        self.pattern_matrix = self.pattern_matrix_full[:, self.col_indices]
        
        # Track candidate solutions
        self._data_loaded = True
    
    def reset_game_state(self):
        """Resets the game state without reloading data."""
        self.candidate_cols = np.arange(len(self.in_superset), dtype=np.int32)
        self.missing_solutions = self.not_in_superset
        self.guess_superset_rows = np.arange(len(self.superset_words), dtype=np.int32)
        self.extra_guess_words = self.not_in_superset
        self.idx_to_word = self.superset_words
        self._guesses = []
    
    def restart_game(self):
        """Resets the game state to allow a fresh start without reloading data."""
        self.reset_game_state()
        self.console.print("[bold green]Game restarted![/bold green]")


    def row_index_to_word(self, row_idx):
        """
        If row_idx >= 0 => superset row
        If row_idx < 0 => index into extra_guess_words
        """
        if row_idx>=0:
            return self.idx_to_word[row_idx]
        else:
            i = -1 - row_idx  # e.g. row_idx=-1 => i=0
            return self.extra_guess_words[i]

    def word_to_row_index(self, w):
        if w in self.superset_idx:
            return self.superset_idx[w]
        else:
            try:
                i = self.extra_guess_words.index(w)
                return -1 - i
            except ValueError:
                return None  # unexpected

    def get_solution_word(self, col_idx):
        """
        col_idx => self.test_words_in_matrix[col_idx]
        """
        return self.test_words_in_matrix[col_idx]

    def fallback_pattern(self, guess_idx, col_idx):
        guess_word = self.row_index_to_word(guess_idx)
        ans_word = self.get_solution_word(col_idx)
        return fallback_pattern(guess_word, ans_word)

    def fallback_pattern_missing_solution(self, guess_idx, missing_sol_word):
        """
        If the solution is not in superset columns,
        we fallback to guess_word vs missing_sol_word
        """
        guess_word = self.row_index_to_word(guess_idx)
        return fallback_pattern(guess_word, missing_sol_word)

    ############################################################################
    # Filter logic
    ############################################################################
    def filter_candidates(self, guess_word, observed_pattern):
        """
        Remove from self.candidate_cols any solution that wouldn't yield observed_pattern
        if guess_word was the actual guess.
        Also remove from missing_solutions if they conflict.
        """
        guess_idx = self.word_to_row_index(guess_word)

        # 1) Filter in-superset columns
        new_cols = []
        for c in self.candidate_cols:
            if guess_idx>=0:
                pat = self.pattern_matrix[guess_idx, c]
            else:
                # fallback
                pat = self.fallback_pattern(guess_idx, c)

            if pat == observed_pattern:
                new_cols.append(c)

        self.candidate_cols = np.array(new_cols, dtype=np.int32)

        # 2) Filter missing solutions
        new_missing = []
        for ms in self.missing_solutions:
            # compute pattern
            if guess_idx is not None:
                # fallback
                pat = fallback_pattern(self.row_index_to_word(guess_idx), ms)
                if pat == observed_pattern:
                    new_missing.append(ms)
            else:
                # guess not found anywhere => unexpected scenario
                pass

        self.missing_solutions = new_missing

    ############################################################################
    # Single-step entropy
    ############################################################################
    def compute_distribution(self, guess_idx):
        """
        Build freq array of size 243 for patterns if we guess guess_idx among current solutions
        which are self.candidate_cols + self.missing_solutions
        """
        freq = np.zeros(243, dtype=int)

        # 1) for in-superset solutions
        for c in self.candidate_cols:
            if guess_idx>=0:
                pat = self.pattern_matrix[guess_idx, c]
            else:
                pat = self.fallback_pattern(guess_idx, c)
            freq[pat]+=1

        # 2) for missing solutions
        for ms in self.missing_solutions:
            guess_word = self.row_index_to_word(guess_idx)
            pat = fallback_pattern(guess_word, ms)
            freq[pat]+=1

        return freq

    def pick_guess(self):
        """
        Among all guess rows (superset + extra), pick the guess with highest single-step entropy
        given current candidate set.
        """
        best_score=-1.0
        best_row=None

        # gather all guess rows
        all_guess_rows = list(self.guess_superset_rows)
        for i, ew in enumerate(self.extra_guess_words):
            all_guess_rows.append(-1 - i)

        for g_idx in all_guess_rows:
            freq = self.compute_distribution(g_idx)
            score = shannon_entropy(freq)
            if score> best_score:
                best_score=score
                best_row=g_idx

        return best_row

    ############################################################################
    def get_guess(self, last_feedback):
        if self._manual=='manual':
            return self.console.input("[bold magenta]Your guess[/bold magenta]: ")

        # if no prior guesses
        if not self._guesses:
            guess_row = self.pick_guess()
            guess_word = self.row_index_to_word(guess_row)
            self._guesses.append(guess_word)
            return guess_word

        # parse feedback
        pattern_code = feedback_str_to_code(last_feedback)
        # filter
        last_guess = self._guesses[-1]
        self.filter_candidates(last_guess, pattern_code)

        # if only 1 in-superset solution left + 0 missing => guess that
        if len(self.candidate_cols)==1 and len(self.missing_solutions)==0:
            final_word = self.test_words_in_matrix[self.candidate_cols[0]]
            self._guesses.append(final_word)
            return final_word

        # if no in-superset but 1 missing => guess that
        if len(self.candidate_cols)==0 and len(self.missing_solutions)==1:
            final_word = self.missing_solutions[0]
            self._guesses.append(final_word)
            return final_word

        # else pick next guess
        guess_row = self.pick_guess()
        guess_word = self.row_index_to_word(guess_row)
        self._guesses.append(guess_word)
        return guess_word
