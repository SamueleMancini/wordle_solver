import yaml
from rich.console import Console
import numpy as np
import pickle
import os
from collections import Counter

# Constants
MISS = 0
MISPLACED = 1
EXACT = 2

class Guesser:
    def __init__(self, manual):
        self.all_words = yaml.load(open('dev_wordlist.yaml'), Loader=yaml.FullLoader)
        self.possible_words = list(self.all_words)
        self._manual = manual
        self.console = Console()
        self._tried = []
        self.history = []  # Store previous guesses and their patterns
        
        # Use precomputed data if available
        self.pattern_cache = {}
        
        # Load precomputed best first guesses and pattern data if available
        self.precomputed_file = 'precomputed_data.pkl'
        if os.path.exists(self.precomputed_file):
            try:
                with open(self.precomputed_file, 'rb') as f:
                    precomputed_data = pickle.load(f)
                    self.best_starters = precomputed_data.get('best_starters', [])
                    if 'pattern_cache' in precomputed_data:
                        self.pattern_cache = precomputed_data['pattern_cache']
            except:
                # Fallback if loading fails
                self.best_starters = ['slate', 'crane', 'trace', 'slant', 'crate']
        else:
            # Hardcoded best starters (based on information theory)
            self.best_starters = ['slate', 'crane', 'trace', 'slant', 'crate']

    def restart_game(self):
        self._tried = []
        self.history = []
        self.possible_words = list(self.all_words)

    def compute_pattern(self, guess, answer):
        """Optimized pattern computation between a guess and an answer"""
        cache_key = (guess, answer)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
            
        pattern = [0] * 5
        # Copy answer to track remaining letters
        answer_chars = list(answer)
        
        # First pass: find exact matches
        for i in range(5):
            if guess[i] == answer[i]:
                pattern[i] = EXACT
                answer_chars[i] = None  # Mark as used
        
        # Second pass: find misplaced letters
        for i in range(5):
            if pattern[i] != EXACT and guess[i] in answer_chars:
                pattern[i] = MISPLACED
                # Find and mark the first occurrence of this letter as used
                idx = answer_chars.index(guess[i])
                answer_chars[idx] = None
        
        # Convert pattern to a single integer using base-3 encoding
        code = sum(pattern[i] * (3**i) for i in range(5))
        
        # Cache result
        self.pattern_cache[cache_key] = code
        return code

    def parse_feedback(self, feedback):
        """Convert feedback string to pattern integer"""
        code = 0
        for i, ch in enumerate(feedback):
            if ch == '+':
                digit = MISS
            elif ch == '-':
                digit = MISPLACED
            else:
                digit = EXACT
            code += digit * (3**i)
        return code

    def filter_words(self, guess, pattern_int):
        """Filter possible words based on the feedback pattern"""
        return [word for word in self.possible_words 
                if self.compute_pattern(guess, word) == pattern_int]

    def calculate_entropy(self, guess, possible_words):
        """Calculate the expected information gain for a guess"""
        # Get all possible patterns for this guess
        pattern_counts = Counter()
        for word in possible_words:
            pattern = self.compute_pattern(guess, word)
            pattern_counts[pattern] += 1
        
        # Calculate entropy
        total = len(possible_words)
        entropy = 0
        for count in pattern_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return entropy

    def find_best_guess(self):
        """Find the best guess based on information theory"""
        if len(self.possible_words) == 1:
            return self.possible_words[0]
        
        if len(self.possible_words) == 2:
            # When only two words remain, pick the first one that hasn't been tried
            for word in self.possible_words:
                if word not in self._tried:
                    return word
            # If somehow both have been tried, return the first one
            return self.possible_words[0]
        
        # Define candidate words to evaluate
        candidates = []
        
        # For efficiency, only consider a subset of words when there are many possibilities
        if len(self.possible_words) > 100:
            # Add words that haven't been tried yet from our best starter words
            candidates.extend([w for w in self.best_starters if w not in self._tried])
            
            # Add a random selection from the possible words (prioritizing untried words)
            untried_possible = [w for w in self.possible_words if w not in self._tried]
            if untried_possible:
                sample_size = min(50, len(untried_possible))
                candidates.extend(np.random.choice(untried_possible, sample_size, replace=False))
            
            # If we still don't have enough candidates, add some random words from all_words
            if len(candidates) < 50:
                additional_needed = 50 - len(candidates)
                untried_all = [w for w in self.all_words if w not in self._tried and w not in candidates]
                if untried_all:
                    sample_size = min(additional_needed, len(untried_all))
                    candidates.extend(np.random.choice(untried_all, sample_size, replace=False))
        elif len(self.possible_words) > 2:
            # For a moderate number of possibilities, consider all possible words plus some good starters
            candidates = list(set(self.possible_words) | set(self.best_starters))
            # Filter out already tried words if possible
            untried = [w for w in candidates if w not in self._tried]
            if untried:
                candidates = untried
        else:
            # For very few possibilities, just use those
            candidates = self.possible_words
        
        # Make sure we have at least one candidate
        if not candidates:
            # Fall back to using possible words or all words
            candidates = [w for w in self.possible_words if w not in self._tried]
            if not candidates:
                candidates = [w for w in self.all_words if w not in self._tried]
                if not candidates:
                    # Last resort: just use all possible words
                    candidates = self.possible_words
        
        # Find the word with highest entropy
        best_word = None
        best_entropy = -1
        
        for word in candidates:
            entropy = self.calculate_entropy(word, self.possible_words)
            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word
        
        # In case we still don't have a best word, return the first untried possible word
        if best_word is None:
            for word in self.possible_words:
                if word not in self._tried:
                    return word
            # If everything has been tried, return the first possible word
            return self.possible_words[0] if self.possible_words else self.all_words[0]
        
        return best_word

    def get_guess(self, result):
        '''This function returns your guess as a string.'''
        if self._manual == 'manual':
            return self.console.input('Your guess:\n')
        
        # First guess is always one of our best starters
        if not self._tried:
            guess = self.best_starters[0]  # 'slate' is a good first guess
        else:
            # Parse feedback and filter words
            feedback_int = self.parse_feedback(result)
            last_guess = self._tried[-1]
            
            # Store in history for consistency checks
            self.history.append((last_guess, feedback_int))
            
            # Filter possible words
            self.possible_words = self.filter_words(last_guess, feedback_int)
            
            # Safety check: ensure we have at least one possible word
            if not self.possible_words:
                # If our filtering removed all words, reset to original list
                # (this is a fallback that shouldn't happen with correct logic)
                self.possible_words = [w for w in self.all_words 
                                      if all(self.compute_pattern(g, w) == p 
                                             for g, p in self.history)]
                
                # If still empty (extremely unlikely), reset completely
                if not self.possible_words:
                    self.possible_words = list(self.all_words)
            
            # Get the best next guess
            guess = self.find_best_guess()
            
            # Safety check: make sure we're not reusing a word we've already tried
            attempt_count = 0
            while guess in self._tried and attempt_count < 10:
                # If we somehow picked a word we've already tried, find another one
                # (this shouldn't happen with the updated logic, but just in case)
                untried = [w for w in self.possible_words if w not in self._tried]
                if untried:
                    guess = untried[0]
                else:
                    # If all possible words have been tried, pick a random untried word
                    untried_all = [w for w in self.all_words if w not in self._tried]
                    if untried_all:
                        guess = np.random.choice(untried_all)
                    else:
                        # As a last resort, pick a random word
                        guess = np.random.choice(self.all_words)
                attempt_count += 1
        
        self._tried.append(guess)
        self.console.print(guess)
        return guess