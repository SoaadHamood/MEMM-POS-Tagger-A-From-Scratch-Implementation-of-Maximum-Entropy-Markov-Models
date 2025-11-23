from scipy import sparse
from collections import OrderedDict, defaultdict
import numpy as np
from typing import List, Dict, Tuple, Set
import re
from sklearn.model_selection import KFold


WORD = 0
TAG = 1




class FeatureStatistics:
   def __init__(self):
       self.n_total_features = 0  # Total number of features accumulated


       # Init all features dictionaries - expanded feature list with specialized features
       feature_dict_list = [
           "f100", "f101", "f102", "f103", "f104", "f105", "f106", "f107",
           "num", "capital",
           # New specialized feature classes
           "currency", "direction", "title", "organization", "that_features",
           "compound_noun", "context_specific"
       ]
       self.feature_rep_dict = {fd: OrderedDict() for fd in feature_dict_list}


       self.tags = set()  # a set of all the seen tags
       self.tags.add("~")
       self.tags_counts = defaultdict(int)  # counts for each tag
       self.words_count = defaultdict(int)  # counts for each word
       self.histories = []  # all histories seen in training


       # Additional statistics for analysis
       self.word_tag_dict = defaultdict(set)  # For each word, store all its possible tags
       self.tag_bigram_counts = defaultdict(int)  # Count tag bigrams
       self.tag_trigram_counts = defaultdict(int)  # Count tag trigrams


       # Known currency terms (to help with "yen" misclassification)
       self.currency_terms = {
           "yen", "dollar", "dollars", "euro", "euros", "pound", "pounds",
           "yuan", "rupee", "rupees", "peso", "pesos", "won", "krona"
       }


       # Directional/temporal words often confused between IN and RB
       self.directional_words = {
           "up", "down", "ago", "away", "back", "out", "off", "on", "in", "about"
       }


       # Common title words
       self.title_words = {
           "chief", "executive", "director", "president", "chairman", "officer",
           "manager", "head", "secretary", "minister", "professor", "dr", "mr", "mrs", "ms"
       }


       # Common organization words
       self.org_words = {
           "department", "bureau", "agency", "committee", "commission", "ministry",
           "institute", "corporation", "securities", "bank", "association"
       }


       # Compound noun suffixes
       self.compound_suffixes = {
           "-based", "-free", "-like", "-related", "-style", "-specific", "-oriented",
           "-out", "-up", "-down", "-in", "-off"
       }


   def _update_feature_count(self, feature_class, feature_key):
       """Helper method to update feature counts"""
       if feature_class not in self.feature_rep_dict:
           return  # Skip if feature class is not defined


       if feature_key not in self.feature_rep_dict[feature_class]:
           self.feature_rep_dict[feature_class][feature_key] = 1
       else:
           self.feature_rep_dict[feature_class][feature_key] += 1


   def get_word_tag_pair_count(self, file_path) -> None:
       """
       Extract out of text all word/tag pairs and feature statistics
       @param: file_path: full path of the file to read
       Updates the histories list
       """
       with open(file_path) as file:
           for line in file:
               if line[-1:] == "\n":
                   line = line[:-1]
               split_words = line.split(' ')


               # Create the sentence with context symbols
               sentence = [("*", "*"), ("*", "*")]
               for pair in split_words:
                   sentence.append(tuple(pair.split("_")))
               sentence.append(("~", "~"))


               # Extract features from the sentence
               for i in range(2, len(sentence) - 1):
                   cur_word, cur_tag = sentence[i]
                   p_word, p_tag = sentence[i - 1]
                   pp_word, pp_tag = sentence[i - 2]
                   n_word = sentence[i + 1][0] if i + 1 < len(sentence) else "~"
                   n_tag = sentence[i + 1][1] if i + 1 < len(sentence) else "~"
                   # Get the second next word and tag
                   nn_word = sentence[i + 2][0] if i + 2 < len(sentence) else "~"
                   nn_tag = sentence[i + 2][1] if i + 2 < len(sentence) else "~"


                   # Basic statistics
                   self.tags.add(cur_tag)
                   self.tags_counts[cur_tag] += 1
                   self.words_count[cur_word] += 1
                   self.word_tag_dict[cur_word].add(cur_tag)


                   # Update tag n-gram statistics
                   self.tag_bigram_counts[(p_tag, cur_tag)] += 1
                   self.tag_trigram_counts[(pp_tag, p_tag, cur_tag)] += 1


                   # Original features
                   # f100: current word, current tag
                   self._update_feature_count("f100", (cur_word, cur_tag))


                   # f101: previous word, current tag
                   self._update_feature_count("f101", (p_word, cur_tag))


                   # f102: previous previous word, current tag
                   self._update_feature_count("f102", (pp_word, cur_tag))


                   # f103: next word, current tag
                   self._update_feature_count("f103", (n_word, cur_tag))


                   # f104: previous tag, current tag
                   self._update_feature_count("f104", (p_tag, cur_tag))


                   # f105: previous previous tag, current tag
                   self._update_feature_count("f105", (pp_tag, cur_tag))


                   # f106: previous tag + previous previous tag, current tag
                   self._update_feature_count("f106", ((pp_tag, p_tag), cur_tag))


                   # f107: Prefixes and suffixes
                   # Prefixes (first 1-4 characters)
                   for prefix_len in range(1, min(5, len(cur_word) + 1)):
                       prefix = cur_word[:prefix_len]
                       self._update_feature_count("f107", ((prefix, "prefix"), cur_tag))


                   # Suffixes (last 1-4 characters)
                   for suffix_len in range(1, min(5, len(cur_word) + 1)):
                       suffix = cur_word[-suffix_len:]
                       self._update_feature_count("f107", ((suffix, "suffix"), cur_tag))


                   # Number features
                   if any(char.isdigit() for char in cur_word):
                       self._update_feature_count("num", ("contains_digit", cur_tag))
                   if cur_word.isdigit():
                       self._update_feature_count("num", ("is_number", cur_tag))


                   # Capital letter features
                   if cur_word and cur_word[0].isupper():
                       self._update_feature_count("capital", ("starts_capital", cur_tag))
                   if cur_word.isupper() and len(cur_word) > 1:
                       self._update_feature_count("capital", ("all_caps", cur_tag))


                   # NEW SPECIALIZED FEATURES


                   # 1. Currency features - address the "yen" issue
                   cur_lower = cur_word.lower()
                   if cur_lower in self.currency_terms:
                       self._update_feature_count("currency", ("is_currency", cur_tag))


                       # Check previous words for numbers or money amounts
                       if p_word.isdigit() or p_word == "billion" or p_word == "million" or p_word == "trillion":
                           self._update_feature_count("currency", (("number_before_currency", cur_lower), cur_tag))


                       # Specifically address "billion yen" pattern
                       if p_word == "billion" and cur_lower == "yen":
                           self._update_feature_count("currency", ("billion_yen", cur_tag))


                   # 2. Directional/temporal words features - address "ago", "up", "down" issues
                   if cur_lower in self.directional_words:
                       self._update_feature_count("direction", ("is_directional", cur_tag))


                       # Specific patterns for "ago"
                       if cur_lower == "ago":
                           if p_word == "year" or p_word == "years" or p_word == "month" or p_word == "months":
                               self._update_feature_count("direction", ("time_ago", cur_tag))


                       # Special features for "up" and "down"
                       if cur_lower in ["up", "down"]:
                           # If preceded by certain verbs
                           if p_tag.startswith("VB"):
                               self._update_feature_count("direction", ((p_tag, cur_lower), cur_tag))


                           # If followed by preposition
                           if n_tag == "IN":
                               self._update_feature_count("direction", ((cur_lower, "before_IN"), cur_tag))


                   # 3. Title word features - address "chief executive" issues
                   if cur_lower in self.title_words:
                       self._update_feature_count("title", ("is_title_word", cur_tag))


                       # Look for title + position patterns
                       if cur_lower == "chief" and n_word.lower() == "executive":
                           self._update_feature_count("title", ("chief_executive", cur_tag))


                   # 4. Organization features - address "Department" etc.
                   if cur_lower in self.org_words or cur_word in self.org_words:
                       self._update_feature_count("organization", ("is_org_word", cur_tag))


                       # If capitalized organization word
                       if cur_word[0].isupper() and cur_lower in self.org_words:
                           self._update_feature_count("organization", ("capital_org_word", cur_tag))


                       # If followed by "of"
                       if n_word.lower() == "of":
                           self._update_feature_count("organization", ((cur_lower, "before_of"), cur_tag))


                   # 5. Special "that" features
                   if cur_lower == "that":
                       self._update_feature_count("that_features", ("is_that", cur_tag))


                       # Check if followed by verb (possible WDT)
                       if n_tag.startswith("VB"):
                           self._update_feature_count("that_features", ("that_before_verb", cur_tag))


                       # Check if followed by noun (possible DT)
                       if n_tag.startswith("NN"):
                           self._update_feature_count("that_features", ("that_before_noun", cur_tag))


                       # Check if preceded by preposition (possible IN)
                       if p_tag == "IN":
                           self._update_feature_count("that_features", ("in_before_that", cur_tag))


                       # Check for "so that" or "such that" patterns
                       if p_word.lower() in ["so", "such"]:
                           self._update_feature_count("that_features", ((p_word.lower(), "that"), cur_tag))


                   # 6. Compound noun features - for hyphenated words and compound terms
                   if "-" in cur_word:
                       self._update_feature_count("compound_noun", ("has_hyphen", cur_tag))


                       # Check for common compound suffixes
                       for suffix in self.compound_suffixes:
                           if cur_word.endswith(suffix):
                               self._update_feature_count("compound_noun", ((suffix, "compound"), cur_tag))


                   # 7. Context-specific features for common error cases


                   # Wider context features (3-grams)
                   self._update_feature_count("context_specific",
                                              ((p_word.lower(), cur_word.lower(), n_word.lower()), cur_tag))


                   # Currency amount context
                   if p_word.isdigit() or p_word in ["billion", "million", "trillion"]:
                       self._update_feature_count("context_specific", (("amount", cur_word.lower()), cur_tag))


                   # Specific feature for plural proper nouns (NNPS) vs plural nouns (NNS)
                   if cur_word.endswith("s") and cur_word[0].isupper():
                       self._update_feature_count("context_specific", ("capital_plural", cur_tag))


                   # Add history
                   history = (cur_word, cur_tag, p_word, p_tag, pp_word, pp_tag, n_word)
                   self.histories.append(history)


   def analyze_data(self):
       """
       Analyze the training data to get statistics
       """
       stats = {
           'total_words': sum(self.words_count.values()),
           'vocab_size': len(self.words_count),
           'total_tags': len(self.tags),
           'tag_distribution': sorted([(tag, count) for tag, count in self.tags_counts.items()],
                                      key=lambda x: x[1], reverse=True),
           'avg_tags_per_word': sum(len(tags) for tags in self.word_tag_dict.values()) / len(
               self.word_tag_dict) if self.word_tag_dict else 0,
           'ambiguous_words': sum(1 for tags in self.word_tag_dict.values() if len(tags) > 1),
           'ambiguous_percentage': sum(1 for tags in self.word_tag_dict.values() if len(tags) > 1) / len(
               self.word_tag_dict) if self.word_tag_dict else 0,
       }
       return stats




class Feature2id:
   def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
       """
       @param feature_statistics: the feature statistics object
       @param threshold: the minimal number of appearances a feature should have to be taken
       """
       self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
       self.threshold = threshold  # feature count threshold - empirical count must be higher than this


       self.n_total_features = 0  # Total number of features accumulated


       # Init all features dictionaries - updated to include all feature classes
       self.feature_to_idx = {
           "f100": OrderedDict(),
           "f101": OrderedDict(),
           "f102": OrderedDict(),
           "f103": OrderedDict(),
           "f104": OrderedDict(),
           "f105": OrderedDict(),
           "f106": OrderedDict(),
           "f107": OrderedDict(),
           "num": OrderedDict(),
           "capital": OrderedDict(),
           # New specialized feature classes
           "currency": OrderedDict(),
           "direction": OrderedDict(),
           "title": OrderedDict(),
           "organization": OrderedDict(),
           "that_features": OrderedDict(),
           "compound_noun": OrderedDict(),
           "context_specific": OrderedDict()
       }
       self.represent_input_with_features = OrderedDict()
       self.histories_matrix = OrderedDict()
       self.histories_features = OrderedDict()
       self.small_matrix = sparse.csr_matrix
       self.big_matrix = sparse.csr_matrix


       # Feature cache for faster lookup during inference
       self.feature_cache = {}


   def get_features_idx(self) -> None:
       """
       Assigns each feature that appeared enough time in the train files an idx.
       Saves those indices to self.feature_to_idx
       """
       for feat_class in self.feature_statistics.feature_rep_dict:
           if feat_class not in self.feature_to_idx:
               continue
           for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
               if count >= self.threshold:
                   self.feature_to_idx[feat_class][feat] = self.n_total_features
                   self.n_total_features += 1
       print(f"You have {self.n_total_features} features!")


   def feature_statistics_per_class(self) -> Dict[str, int]:
       """
       Count the number of features per feature class
       """
       counts = {}
       for feat_class in self.feature_to_idx:
           counts[feat_class] = len(self.feature_to_idx[feat_class])
       return counts


   def prune_features(self, pruning_threshold):
       """
       Prune features that appear less than pruning_threshold times
       """
       original_count = self.n_total_features


       # Create new dictionaries to hold the pruned features
       pruned_feature_to_idx = {
           feat_class: OrderedDict() for feat_class in self.feature_to_idx.keys()
       }


       # Reset feature count
       self.n_total_features = 0


       # Add features that meet the pruning threshold
       for feat_class in self.feature_to_idx:
           for feat, _ in self.feature_to_idx[feat_class].items():
               count = self.feature_statistics.feature_rep_dict[feat_class].get(feat, 0)
               if count >= pruning_threshold:
                   pruned_feature_to_idx[feat_class][feat] = self.n_total_features
                   self.n_total_features += 1


       # Replace the old feature_to_idx with the pruned one
       self.feature_to_idx = pruned_feature_to_idx


       # Clear the feature cache
       self.feature_cache = {}


       print(f"Pruned features from {original_count} to {self.n_total_features}")
       return original_count - self.n_total_features


   def calc_represent_input_with_features(self) -> None:
       """
       initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
       """
       big_r = 0
       big_rows = []
       big_cols = []
       small_rows = []
       small_cols = []
       for small_r, hist in enumerate(self.feature_statistics.histories):
           for c in represent_input_with_features(hist, self.feature_to_idx):
               small_rows.append(small_r)
               small_cols.append(c)
           for r, y_tag in enumerate(self.feature_statistics.tags):
               demi_hist = (hist[0], y_tag, hist[2], hist[3], hist[4], hist[5], hist[6])
               self.histories_features[demi_hist] = []
               for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                   big_rows.append(big_r)
                   big_cols.append(c)
                   self.histories_features[demi_hist].append(c)
               big_r += 1
       self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                           shape=(len(self.feature_statistics.tags) * len(
                                               self.feature_statistics.histories), self.n_total_features),
                                           dtype=bool)
       self.small_matrix = sparse.csr_matrix(
           (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
           shape=(len(
               self.feature_statistics.histories), self.n_total_features), dtype=bool)




def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple, int]]) -> List[int]:
   """
   Extract feature vector for a given history
   @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
   @param dict_of_dicts: a dictionary of each feature and the index it was given
   @return a list with all features that are relevant to the given history
   """
   c_word = history[0]
   c_tag = history[1]
   p_word = history[2]
   p_tag = history[3]
   pp_word = history[4]
   pp_tag = history[5]
   n_word = history[6]
   features = []


   # Convert to lowercase for certain feature checks
   c_lower = c_word.lower()
   p_lower = p_word.lower() if p_word else ""
   n_lower = n_word.lower() if n_word else ""


   # Original features
   # f100: current word, current tag
   if "f100" in dict_of_dicts and (c_word, c_tag) in dict_of_dicts["f100"]:
       features.append(dict_of_dicts["f100"][(c_word, c_tag)])


   # f101: previous word, current tag
   if "f101" in dict_of_dicts and (p_word, c_tag) in dict_of_dicts["f101"]:
       features.append(dict_of_dicts["f101"][(p_word, c_tag)])


   # f102: previous previous word, current tag
   if "f102" in dict_of_dicts and (pp_word, c_tag) in dict_of_dicts["f102"]:
       features.append(dict_of_dicts["f102"][(pp_word, c_tag)])


   # f103: next word, current tag
   if "f103" in dict_of_dicts and (n_word, c_tag) in dict_of_dicts["f103"]:
       features.append(dict_of_dicts["f103"][(n_word, c_tag)])


   # f104: previous tag, current tag
   if "f104" in dict_of_dicts and (p_tag, c_tag) in dict_of_dicts["f104"]:
       features.append(dict_of_dicts["f104"][(p_tag, c_tag)])


   # f105: previous previous tag, current tag
   if "f105" in dict_of_dicts and (pp_tag, c_tag) in dict_of_dicts["f105"]:
       features.append(dict_of_dicts["f105"][(pp_tag, c_tag)])


   # f106: previous tag + previous previous tag, current tag
   if "f106" in dict_of_dicts and ((pp_tag, p_tag), c_tag) in dict_of_dicts["f106"]:
       features.append(dict_of_dicts["f106"][((pp_tag, p_tag), c_tag)])


   # f107: prefixes and suffixes
   if "f107" in dict_of_dicts:
       # Prefixes (first 1-4 characters)
       for prefix_len in range(1, min(5, len(c_word) + 1)):
           prefix = c_word[:prefix_len]
           if ((prefix, "prefix"), c_tag) in dict_of_dicts["f107"]:
               features.append(dict_of_dicts["f107"][((prefix, "prefix"), c_tag)])


       # Suffixes (last 1-4 characters)
       for suffix_len in range(1, min(5, len(c_word) + 1)):
           suffix = c_word[-suffix_len:]
           if ((suffix, "suffix"), c_tag) in dict_of_dicts["f107"]:
               features.append(dict_of_dicts["f107"][((suffix, "suffix"), c_tag)])


   # Number features
   if "num" in dict_of_dicts:
       # Check if word contains digits
       has_digit = any(char.isdigit() for char in c_word)
       if has_digit and ("contains_digit", c_tag) in dict_of_dicts["num"]:
           features.append(dict_of_dicts["num"][("contains_digit", c_tag)])


       # Check if word is entirely a number
       is_number = c_word.isdigit()
       if is_number and ("is_number", c_tag) in dict_of_dicts["num"]:
           features.append(dict_of_dicts["num"][("is_number", c_tag)])


   # Capital letters features
   if "capital" in dict_of_dicts:
       # Check if word starts with capital letter
       starts_with_capital = c_word and c_word[0].isupper()
       if starts_with_capital and ("starts_capital", c_tag) in dict_of_dicts["capital"]:
           features.append(dict_of_dicts["capital"][("starts_capital", c_tag)])


       # Check if all letters are uppercase
       is_all_caps = c_word.isupper() and len(c_word) > 1
       if is_all_caps and ("all_caps", c_tag) in dict_of_dicts["capital"]:
           features.append(dict_of_dicts["capital"][("all_caps", c_tag)])


   # NEW SPECIALIZED FEATURES


   # 1. Currency features
   if "currency" in dict_of_dicts:
       # Check if the word is a known currency term
       if c_lower in ["yen", "dollar", "dollars", "euro", "euros", "pound", "pounds", "yuan", "rupee", "rupees",
                      "peso", "pesos", "won", "krona"]:
           if ("is_currency", c_tag) in dict_of_dicts["currency"]:
               features.append(dict_of_dicts["currency"][("is_currency", c_tag)])


           # Check if preceded by a number or quantity term
           if p_word.isdigit() or p_lower in ["billion", "million", "trillion"]:
               if (("number_before_currency", c_lower), c_tag) in dict_of_dicts["currency"]:
                   features.append(dict_of_dicts["currency"][(("number_before_currency", c_lower), c_tag)])


           # Special case for "billion yen"
           if p_lower == "billion" and c_lower == "yen":
               if ("billion_yen", c_tag) in dict_of_dicts["currency"]:
                   features.append(dict_of_dicts["currency"][("billion_yen", c_tag)])


   # 2. Directional word features
   if "direction" in dict_of_dicts:
       # Check if word is a directional/temporal term
       if c_lower in ["up", "down", "ago", "away", "back", "out", "off", "on", "in", "about"]:
           if ("is_directional", c_tag) in dict_of_dicts["direction"]:
               features.append(dict_of_dicts["direction"][("is_directional", c_tag)])


           # Special case for "ago"
           if c_lower == "ago":
               if p_lower in ["year", "years", "month", "months", "day", "days", "week", "weeks"]:
                   if ("time_ago", c_tag) in dict_of_dicts["direction"]:
                       features.append(dict_of_dicts["direction"][("time_ago", c_tag)])


           # Special case for "up" and "down"
           if c_lower in ["up", "down"]:
               # If preceded by verb
               if p_tag and p_tag.startswith("VB"):
                   if ((p_tag, c_lower), c_tag) in dict_of_dicts["direction"]:
                       features.append(dict_of_dicts["direction"][((p_tag, c_lower), c_tag)])


               # If followed by preposition-like term
               if n_word.lower() in ["to", "on", "at", "in", "by", "from"]:
                   if ((c_lower, "before_IN"), c_tag) in dict_of_dicts["direction"]:
                       features.append(dict_of_dicts["direction"][((c_lower, "before_IN"), c_tag)])


   # 3. Title word features
   if "title" in dict_of_dicts:
       # Check if word is a title/position term
       if c_lower in ["chief", "executive", "director", "president", "chairman", "officer",
                      "manager", "head", "secretary", "minister", "professor", "dr", "mr", "mrs", "ms"]:
           if ("is_title_word", c_tag) in dict_of_dicts["title"]:
               features.append(dict_of_dicts["title"][("is_title_word", c_tag)])


           # Check for title + position pattern
           if c_lower == "chief" and n_lower == "executive":
               if ("chief_executive", c_tag) in dict_of_dicts["title"]:
                   features.append(dict_of_dicts["title"][("chief_executive", c_tag)])


   # 4. Organization word features
   if "organization" in dict_of_dicts:
       # Check if word is an organization term
       if c_lower in ["department", "bureau", "agency", "committee", "commission", "ministry",
                      "institute", "corporation", "securities", "bank", "association"]:
           if ("is_org_word", c_tag) in dict_of_dicts["organization"]:
               features.append(dict_of_dicts["organization"][("is_org_word", c_tag)])


           # If capitalized org word
           if c_word and c_word[0].isupper():
               if ("capital_org_word", c_tag) in dict_of_dicts["organization"]:
                   features.append(dict_of_dicts["organization"][("capital_org_word", c_tag)])


           # If followed by "of"
           if n_lower == "of":
               if ((c_lower, "before_of"), c_tag) in dict_of_dicts["organization"]:
                   features.append(dict_of_dicts["organization"][((c_lower, "before_of"), c_tag)])


   # 5. Special "that" features
   if "that_features" in dict_of_dicts and c_lower == "that":
       if ("is_that", c_tag) in dict_of_dicts["that_features"]:
           features.append(dict_of_dicts["that_features"][("is_that", c_tag)])


       # Check context patterns for "that"
       if p_lower in ["so", "such"]:
           if ((p_lower, "that"), c_tag) in dict_of_dicts["that_features"]:
               features.append(dict_of_dicts["that_features"][((p_lower, "that"), c_tag)])


   # 6. Compound noun features
   if "compound_noun" in dict_of_dicts and "-" in c_word:
       if ("has_hyphen", c_tag) in dict_of_dicts["compound_noun"]:
           features.append(dict_of_dicts["compound_noun"][("has_hyphen", c_tag)])


       # Check for specific compound suffixes
       for suffix in ["-based", "-free", "-like", "-related", "-style", "-specific", "-oriented",
                      "-out", "-up", "-down", "-in", "-off"]:
           if c_word.endswith(suffix):
               if ((suffix, "compound"), c_tag) in dict_of_dicts["compound_noun"]:
                   features.append(dict_of_dicts["compound_noun"][((suffix, "compound"), c_tag)])


   # 7. Context-specific features
   if "context_specific" in dict_of_dicts:
       # Capture 3-word context patterns
       if ((p_lower, c_lower, n_lower), c_tag) in dict_of_dicts["context_specific"]:
           features.append(dict_of_dicts["context_specific"][((p_lower, c_lower, n_lower), c_tag)])


       # Currency amount pattern
       if p_lower.isdigit() or p_lower in ["billion", "million", "trillion"]:
           if (("amount", c_lower), c_tag) in dict_of_dicts["context_specific"]:
               features.append(dict_of_dicts["context_specific"][(("amount", c_lower), c_tag)])


       # Special case for capitalized plurals (to help with NNPS vs NNS confusion)
       if c_word and c_word[0].isupper() and c_word.endswith("s"):
           if ("capital_plural", c_tag) in dict_of_dicts["context_specific"]:
               features.append(dict_of_dicts["context_specific"][("capital_plural", c_tag)])


   return features




def preprocess_train(train_path, threshold):
   """
   Preprocess the training data
   """
   # Statistics
   statistics = FeatureStatistics()
   statistics.get_word_tag_pair_count(train_path)


   # Analyze the data
   data_stats = statistics.analyze_data()
   print("\n=== Data Analysis ===")
   print(f"Total words: {data_stats['total_words']}")
   print(f"Vocabulary size: {data_stats['vocab_size']}")
   print(f"Number of unique tags: {data_stats['total_tags']}")
   print(f"Top 5 most common tags: {data_stats['tag_distribution'][:5]}")
   print(f"Average tags per word: {data_stats['avg_tags_per_word']:.2f}")
   print(f"Ambiguous words: {data_stats['ambiguous_words']} ({data_stats['ambiguous_percentage'] * 100:.2f}%)")
   print("=====================\n")


   # feature2id
   feature2id = Feature2id(statistics, threshold)
   feature2id.get_features_idx()
   feature2id.calc_represent_input_with_features()


   # Print feature statistics
   feature_stats = feature2id.feature_statistics_per_class()
   print("\n=== Feature Statistics ===")
   for feat_class, count in feature_stats.items():
       print(f"{feat_class}: {count} features")
   print("==========================\n")


   print(f"Total features: {feature2id.n_total_features}")


   return statistics, feature2id




def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
   """
   reads a test file
   @param file_path: the path to the file
   @param tagged: whether the file is tagged (validation set) or not (test set)
   @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
   """
   list_of_sentences = []
   with open(file_path) as f:
       for line in f:
           if line[-1:] == "\n":
               line = line[:-1]
           sentence = (["*", "*"], ["*", "*"])
           split_words = line.split(' ')
           for word_idx in range(len(split_words)):
               if tagged:
                   cur_word, cur_tag = split_words[word_idx].split('_')
               else:
                   cur_word, cur_tag = split_words[word_idx], ""
               sentence[WORD].append(cur_word)
               sentence[TAG].append(cur_tag)
           sentence[WORD].append("~")
           sentence[TAG].append("~")
           list_of_sentences.append(sentence)
   return list_of_sentences




def cross_validate(train_data, k=5, threshold=1, lam=1):
   """
   Perform k-fold cross-validation on the training data
   Useful for Model 2 which has no test data


   @param train_data: List of sentences in the training data
   @param k: Number of folds
   @param threshold: Feature threshold
   @param lam: Lambda regularization parameter
   @return: Average accuracy across folds
   """
   from sklearn.model_selection import KFold
   import pickle
   import tempfile
   from optimization import get_optimal_vector


   # Prepare the data for cross-validation
   kf = KFold(n_splits=k, shuffle=True, random_state=42)
   fold_accuracies = []


   # Create temporary files for each fold
   with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_train, \
           tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_test, \
           tempfile.NamedTemporaryFile(mode='wb+', delete=False) as temp_weights:


       temp_train_path = temp_train.name
       temp_test_path = temp_test.name
       temp_weights_path = temp_weights.name


       # Perform k-fold cross-validation
       for fold, (train_idx, test_idx) in enumerate(kf.split(train_data)):
           print(f"=== Fold {fold + 1}/{k} ===")


           # Create training and test files for this fold
           temp_train.seek(0)
           temp_train.truncate()
           for idx in train_idx:
               temp_train.write(train_data[idx] + '\n')
           temp_train.flush()


           temp_test.seek(0)
           temp_test.truncate()
           for idx in test_idx:
               temp_test.write(train_data[idx] + '\n')
           temp_test.flush()


           # Train model on this fold
           statistics, feature2id = preprocess_train(temp_train_path, threshold)
           get_optimal_vector(statistics, feature2id, lam, temp_weights_path)


           # Test model on this fold
           with open(temp_weights_path, 'rb') as f:
               optimal_params, feature2id = pickle.load(f)
           pre_trained_weights = optimal_params[0]


           # Calculate accuracy
           from inference import tag_all_test, calculate_accuracy
           test_predictions_path = "temp_predictions.wtag"
           tag_all_test(temp_test_path, pre_trained_weights, feature2id, test_predictions_path)
           accuracy = calculate_accuracy(temp_test_path, test_predictions_path)


           print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
           fold_accuracies.append(accuracy)


       # Clean up temporary files
       import os
       os.unlink(temp_train_path)
       os.unlink(temp_test_path)
       os.unlink(temp_weights_path)
       if os.path.exists("temp_predictions.wtag"):
           os.unlink("temp_predictions.wtag")


   # Return average accuracy
   avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
   print(f"Average accuracy across {k} folds: {avg_accuracy:.4f}")
   return avg_accuracy

