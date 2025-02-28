from random import choice
import yaml
from rich.console import Console
import pandas as pd
import numpy as np
from collections import Counter
import itertools as it



# def simulate_feedback(true_word, guess):
#     """
#     Simulate Wordle feedback for a guess given the true_word.
    
#     Returns a feedback string where:
#       - Correct letter in the correct position: the letter itself.
#       - Letter in the word but wrong position: '-'
#       - Letter not in the word: '+'
#     """
#     counts = Counter(true_word)
#     results = []
    
#     # First pass: mark correct positions and reduce counts.
#     for i, letter in enumerate(guess):
#         if letter == true_word[i]:
#             results.append(letter)
#             counts[letter] -= 1
#         else:
#             results.append('+')  # placeholder for now
    
#     # Second pass: mark misplaced letters where applicable.
#     for i, letter in enumerate(guess):
#         if letter != true_word[i] and letter in true_word and counts[letter] > 0:
#             results[i] = '-'
#             counts[letter] -= 1

#     return ''.join(results)

# def vectorized_feedback(candidates, guess):
#     """
#     Compute feedback for many candidate words at once.
    
#     Parameters:
#       candidates: a NumPy array of shape (n,) with candidate words (each length 5)
#       guess: a string of length 5
    
#     Returns:
#       A NumPy array of feedback strings (length n).
#     """
#     n = candidates.shape[0]
#     # Split each candidate into an array of characters, shape (n, 5)
#     cand_chars = np.array([list(word) for word in candidates])
#     # Convert the guess into a 1D array of characters, shape (5,)
#     guess_chars = np.array(list(guess))
    
#     # Initialize feedback array with '+' (as default)
#     feedback = np.full((n, 5), '+', dtype='<U1')
    
#     # Compute which positions are exactly correct.
#     correct_mask = (cand_chars == guess_chars)
#     # Broadcast the guess across all candidates:
#     tile_guess = np.tile(guess_chars, (n, 1))
#     # Where correct, assign the guess letter.
#     feedback[correct_mask] = tile_guess[correct_mask]
    
#     # For each candidate word (each row) process misplaced letters.
#     for i in range(n):
#         # Build counts for letters in candidate that were NOT matched exactly.
#         counts = {}
#         for j in range(5):
#             if not correct_mask[i, j]:
#                 letter = cand_chars[i, j]
#                 counts[letter] = counts.get(letter, 0) + 1
#         # For each unmatched position in the guess, if the candidate still has that letter available, mark it as misplaced.
#         for j in range(5):
#             if not correct_mask[i, j]:
#                 letter = guess_chars[j]
#                 if counts.get(letter, 0) > 0:
#                     feedback[i, j] = '-'
#                     counts[letter] -= 1
#     # Join the feedback characters for each candidate into a string.
#     feedback_str = np.apply_along_axis(''.join, 1, feedback)
#     return feedback_str

# def parse_result(guess, result, fixed_positions, misplaced_letters, min_letter_counts, absent_letters):
#     letter_occurrences = {}

#     for i, (g, f) in enumerate(zip(guess, result)):
#         letter_occurrences[g] = letter_occurrences.get(g, 0) + 1
#         if f == g:
#             fixed_positions[i] = g
            
#             min_letter_counts[g] = max(min_letter_counts.get(g, 0), guess.count(g))
    
#     for i, (g, f) in enumerate(zip(guess, result)):
#         if f == '-':
#             misplaced_letters.setdefault(g, set()).add(i)
#             min_letter_counts[g] = max(min_letter_counts.get(g, 0), guess.count(g) - result.count('+'))

#         elif f == '+':
#             if g not in fixed_positions and g not in misplaced_letters:
#                 absent_letters.add(g)

#     return fixed_positions, misplaced_letters, min_letter_counts, absent_letters


# def expected_information_vectorized(guess, candidate_words, cache, sample_candidates=None):
#     """
#     Compute the expected information (in bits) for a given guess using a vectorized feedback calculation.
    
#     Parameters:
#       guess: a candidate guess word (string)
#       candidate_words: list of candidate words (strings)
#       cache: a dictionary for caching feedback results.
#       sample_candidates: if provided, a NumPy array of a sample of candidate words to speed up computation.
    
#     Returns:
#       The expected information gain (entropy in bits) for this guess.
#     """
#     if sample_candidates is None:
#         sample_candidates = np.array(candidate_words)
#     else:
#         sample_candidates = np.array(sample_candidates)
        
#     # Compute feedback for all sampled candidate words for this guess.
#     sample_feedback = vectorized_feedback(sample_candidates, guess)
#     # Update the cache for these pairs.
#     for word, fb in zip(sample_candidates, sample_feedback):
#         cache[(guess, word)] = fb
        
#     # Now, count the frequency of each unique feedback.
#     unique, counts = np.unique(sample_feedback, return_counts=True)
#     total = len(sample_candidates)
#     info_gain = np.sum((counts / total) * np.log2(total / counts))
#     return info_gain


# def best_guess_vectorized(candidate_words, possible_guesses, cache, sample_size=4500):
#     """
#     From possible_guesses, select the guess with the highest expected information gain,
#     using a sample (if candidate_words is large) to approximate the feedback distribution.
    
#     Parameters:
#       candidate_words: list of candidate words (strings) that are still possible.
#       possible_guesses: list of words from which we choose the guess (here, we use candidate_words).
#       cache: a dictionary for caching feedback results.
#       sample_size: if len(candidate_words) > sample_size, sample this many words for the entropy calculation.
    
#     Returns:
#       A tuple (best_guess, info_gain)
#     """
#     candidate_words_arr = np.array(candidate_words)
#     if len(candidate_words) > sample_size:
#         sample_candidates = np.random.choice(candidate_words_arr, sample_size, replace=False)
#     else:
#         sample_candidates = candidate_words_arr

#     best = None
#     best_info = -float('inf')
#     for guess in possible_guesses:
#         info = expected_information_vectorized(guess, candidate_words, cache, sample_candidates)
#         if info > best_info:
#             best_info = info
#             best = guess
#     return best, best_info

MISS = np.uint8(0)
MISPLACED = np.uint8(1)
EXACT = np.uint8(2)


def words_to_int_arrays(words):
    """
    Convert a list of words into a 2D numpy array (shape: [n_words, word_length])
    where each element is the ASCII code (uint8) of that letter.
    """
    return np.array([[ord(c) for c in word] for word in words], dtype=np.uint8)


def convert_feedback(feedback):
    """
    Convert feedback given as a string (e.g., "c-++e") into an integer using base‑3 encoding.
    For each position i:
      - If feedback[i] is '+', assign 0 (MISS).
      - If feedback[i] is '-', assign 1 (MISPLACED).
      - Otherwise (if it’s a letter), assign 2 (EXACT).
    
    The integer is computed as sum(digit * (3**i)) for i in range(word_length).
    
    Parameters:
      feedback: a string of length 5 (like "c-++e")
      guess: the guess string (e.g., "crate") used for reference.
             (In this simple conversion, we assume that if feedback[i] is not '+' or '-', 
              it means the letter is exactly correct.)
    
    Returns:
      An integer in the range 0 to 3^5-1.
    """
    code = 0
    for i, ch in enumerate(feedback):
        if ch == '+':
            digit = 0
        elif ch == '-':
            digit = 1
        else:
            # If it's a letter (e.g., "c" or "e"), we assume it means EXACT.
            digit = 2
        code += digit * (3 ** i)
    return code


def compute_pattern(guess, answer):
    """Compute the Wordle pattern for one guess/answer pair.
    
    Returns an integer (0–242) computed as sum(digit * (3**i))
    for i in range(word_length), where the digit is:
      0 for MISS, 1 for MISPLACED, and 2 for EXACT.
    """
    nl = len(guess)
    pattern = [0] * nl
    # Create a mutable list of answer letters
    answer_letters = list(answer)
    
    # First pass: mark exact (green) matches
    for i in range(nl):
        if guess[i] == answer[i]:
            pattern[i] = 2
            answer_letters[i] = None  # remove that letter so it can't be used again
    
    # Second pass: mark misplaced (yellow) matches
    for i in range(nl):
        if pattern[i] == 0 and guess[i] in answer_letters:
            pattern[i] = 1
            # Remove the first occurrence of the letter
            answer_letters[answer_letters.index(guess[i])] = None
            
    # Convert pattern list to an integer (base 3)
    code = sum(pattern[i] * (3 ** i) for i in range(nl))
    return code

def generate_pattern_matrix(words1, words2):
    """Generate a pattern matrix using the compute_pattern function."""
    nw1 = len(words1)
    nw2 = len(words2)
    matrix = np.zeros((nw1, nw2), dtype=np.uint8)
    for i in range(nw1):
        for j in range(nw2):
            matrix[i, j] = compute_pattern(words1[i], words2[j])
    return matrix


def get_pattern_distributions(allowed_words, possible_words):
    """
    For each allowed guess in allowed_words, compute the probability distribution
    over the 243 possible feedback patterns, assuming each possible answer in possible_words
    is equally likely.
    Returns an array of shape (n_allowed, 243) where each row sums to 1.
    """
    pattern_matrix = generate_pattern_matrix(allowed_words, possible_words)
    n_allowed = pattern_matrix.shape[0]
    distribution = np.zeros((n_allowed, 3**5))
    for i in range(n_allowed):
        counts = np.bincount(pattern_matrix[i], minlength=3**5)
        total = counts.sum()
        if total == 0:
            distribution[i] = np.zeros(3**5)
        else:
            distribution[i] = counts / float(total)
    return distribution


def entropy(distribution):
    """
    Compute the Shannon entropy (in bits) of a probability distribution.
    """
    mask = distribution > 0
    return -np.sum(distribution[mask] * np.log2(distribution[mask]))


def get_entropies(allowed_words, possible_words):
    """
    For each allowed word, compute its expected information (entropy in bits)
    if used as a guess against all possible answers in possible_words.
    Returns a numpy array of entropy values.
    """
    dists = get_pattern_distributions(allowed_words, possible_words)
    return np.array([entropy(dist) for dist in dists])


def get_possible_words(guess, pattern, word_list):
    """
    Filter the list of possible words: return only those that, when compared with the guess,
    yield the given feedback pattern.
    
    Here, 'pattern' is an integer (0 to 3^5-1) encoding the feedback.
    """
    # Compute the pattern between the guess and every word in word_list.
    pattern_matrix = generate_pattern_matrix([guess], word_list)
    return [word for word, p in zip(word_list, pattern_matrix[0]) if p == pattern]


def is_consistent(candidate, history):
    """
    Check whether a candidate word is consistent with all previous (guess, feedback) pairs.
    history is a list of tuples: (guess, feedback_int)
    """
    for guess, feedback_int in history:
        computed = generate_pattern_matrix([guess], [candidate])[0, 0]
        if computed != feedback_int:
            return False
    return True


def filter_candidates(history, candidate_words):
    """
    Filter candidate words to only those that are consistent with the entire history.
    """
    return [word for word in candidate_words if is_consistent(word, history)]


class Guesser:
    '''
        INSTRUCTIONS: This function should return your next guess. 
        Currently it picks a random word from wordlist and returns that.
        You will need to parse the output from Wordle:
        - If your guess contains that character in a different position, Wordle will return a '-' in that position.
        - If your guess does not contain thta character at all, Wordle will return a '+' in that position.
        - If you guesses the character placement correctly, Wordle will return the character. 

        You CANNOT just get the word from the Wordle class, obviously :)
    '''
    def __init__(self, manual):
        self.word_list = pd.read_csv("wordlist.tsv", sep="\t")
        # self.all_words = self.word_list['word'].tolist()
        # self.candidate_words = self.all_words.copy()
        self.all_words = yaml.load(open('combined.yaml'), Loader=yaml.FullLoader)
        # self.possible_words = [word for word in self.all_words]
        self.possible_words = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)

        self._manual = manual
        self.console = Console()
        self.history = []
        self._tried = []
        # self.fixed_positions = [None] * 5
        # self.misplaced_letters = {}
        # self.min_letter_counts = {}
        # self.absent_letters = set()

        # self._feedback_cache = {}


    def restart_game(self):
        self._tried = []
        self.history = []
        self.possible_words = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
        # self.fixed_positions = [None] * 5
        # self.misplaced_letters = {}
        # self.min_letter_counts = {}
        # self.absent_letters = set()
        # self.candidate_words = self.all_words.copy()

        # self._feedback_cache = {}
        


    def get_guess(self, result):
        '''
        This function must return your guess as a string. 
        '''
        if self._manual=='manual':
            return self.console.input('Your guess:\n')
        
        if not self._tried:
            guess = "slate"
            test_pattern = generate_pattern_matrix(["neale"], ["della"])[0, 0]
            
        else:
            result = convert_feedback(result)
            self.history.append((self._tried[-1], result))
            self.possible_words = filter_candidates(self.history, self.possible_words)
            last_guess = self._tried[-1]
            # Filter the candidate list using the feedback pattern.
            
            # print("result:", result)
            # print("history:", self.history)
            self.possible_words = get_possible_words(last_guess, result, self.possible_words)
            if len(self.possible_words) == 1:
                guess = self.possible_words[0]
            else:
                # print("possible words:", self.possible_words)
                # print("pattern matrix:", generate_pattern_matrix([last_guess], self.possible_words))
                # Recompute expected entropies for all allowed guesses against the remaining candidates.
                entropies = get_entropies(self.all_words, self.possible_words)
                guess = self.all_words[np.argmax(entropies)]
        
        self._tried.append(guess)
        self.console.print(guess)
        print(guess)
        return guess
        
        # else:
        #     if self._tried:
        #         last_guess = self._tried[-1]
                
        #         self.fixed_positions, self.misplaced_letters, self.min_letter_counts, self.absent_letters = \
        #             parse_result(last_guess, result, self.fixed_positions, self.misplaced_letters, 
        #                          self.min_letter_counts, self.absent_letters)

        #         prior_probs = self.word_list.set_index('word')['probability'].to_dict()
        #         new_probs = {}
        #         total_prob = 0.0
                
        #         for word in self.word_list['word']:
        #             # Use cached simulation if available.
        #             key = (last_guess, word)
        #             if key not in self._feedback_cache:
        #                 self._feedback_cache[key] = simulate_feedback(word, last_guess)
        #             simulated = self._feedback_cache[key]
                    
        #             likelihood = 1.0 if simulated == result else 0.0
        #             new_probs[word] = prior_probs[word] * likelihood
        #             total_prob += new_probs[word]
                
        #         if total_prob > 0:
        #             for word in new_probs:
        #                 new_probs[word] /= total_prob
        #         else:
        #             raise ValueError("No candidate words match the observed feedback. Check your feedback simulation!")
                
        #         self.word_list['probability'] = self.word_list['word'].map(new_probs)
            
            
        #     Filter candidate words: only those with a nonzero probability.
        #     candidate_df = self.word_list[self.word_list['probability'] > 0]
        #     candidate_words = candidate_df['word'].tolist()
            
        #     For speed, consider restricting possible guesses to candidate words only.
        #     possible_guesses = candidate_words
            
        #     Select the guess with maximal expected information gain.
        #     guess, info_gain = best_guess(candidate_words, possible_guesses, self._feedback_cache)
            
            # self._tried.append(guess)
            # self.console.print(guess)
            # print(guess)


            # return guess



        