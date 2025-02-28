import yaml
from rich.console import Console
import numpy as np


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
    answer_letters = list(answer)
    
    for i in range(nl):
        if guess[i] == answer[i]:
            pattern[i] = 2
            answer_letters[i] = None  
    
    for i in range(nl):
        if pattern[i] == 0 and guess[i] in answer_letters:
            pattern[i] = 1
            answer_letters[answer_letters.index(guess[i])] = None
            
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
        self.all_words = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
        self.possible_words = [word for word in self.all_words]

        self._manual = manual
        self.console = Console()
        self.history = []
        self._tried = []

    def restart_game(self):
        self._tried = []
        self.history = []
        self.possible_words = [word for word in self.all_words]

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
            
            # print("result:", result)
            # print("history:", self.history)
            self.possible_words = get_possible_words(last_guess, result, self.possible_words)
            if len(self.possible_words) == 1:
                guess = self.possible_words[0]
            else:
                # print("possible words:", self.possible_words)
                # print("pattern matrix:", generate_pattern_matrix([last_guess], self.possible_words))
                entropies = get_entropies(self.all_words, self.possible_words)
                guess = self.all_words[np.argmax(entropies)]
        
        self._tried.append(guess)
        self.console.print(guess)
        print(guess)
        return guess
        


        