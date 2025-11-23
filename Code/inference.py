from preprocessing import read_test, represent_input_with_features
import numpy as np
from tqdm import tqdm
import math


def calc_q_prob(history, tag, pre_trained_weights, feature2id):
    """
    Calculate q(y|x) - the probability of tag given history
    Uses the pre-trained weights and features
    """
    # Use feature cache if available
    cache_key = (history[0], tag, history[2], history[3], history[4], history[5], history[6])
    if hasattr(feature2id, 'feature_cache') and cache_key in feature2id.feature_cache:
        features = feature2id.feature_cache[cache_key]
    else:
        # Create history with the given tag
        tagged_history = (history[0], tag, history[2], history[3], history[4], history[5], history[6])
        # Get features for this history-tag pair
        features = represent_input_with_features(tagged_history, feature2id.feature_to_idx)
        # Cache the features if caching is available
        if hasattr(feature2id, 'feature_cache'):
            feature2id.feature_cache[cache_key] = features

    # Calculate linear term (dot product of features and weights)
    score = 0
    for feat_id in features:
        score += pre_trained_weights[feat_id]

    return score


def memm_viterbi(sentence, pre_trained_weights, feature2id):
    """
    MEMM Viterbi implementation for POS tagging with beam search optimization
    @param sentence: a list of words to tag
    @param pre_trained_weights: weights vector for the features
    @param feature2id: feature to ID mapping
    @return: a list of predicted tags for the sentence
    """
    # Get all possible tags (exclude special tags)
    all_tags = list(feature2id.feature_statistics.tags)
    tags = [tag for tag in all_tags if tag not in ['*', '~']]

    n = len(sentence)

    # Settings for beam search
    beam_size = 5  # Adjust based on performance needs
    probability_threshold = -50.0  # Log probability threshold for early pruning

    # Initialize viterbi table and backpointers
    # viterbi[k][u] = max probability of tag sequence ending with tag u at position k
    # bp[k][u] = previous tag that gives the max probability for tag u at position k
    viterbi = [{} for _ in range(n)]
    bp = [{} for _ in range(n)]

    # Base case: position 2 (first actual word)
    viterbi[2] = {}
    bp[2] = {}

    for u in tags:
        # History for the first actual word
        history = (sentence[2], None, sentence[1], '*', sentence[0], '*', sentence[3])

        # Get log probability
        score = calc_q_prob(history, u, pre_trained_weights, feature2id)
        viterbi[2][u] = score
        bp[2][u] = '*'  # Previous tag is the start symbol

    # Handle case where normalization might be needed
    if viterbi[2]:  # Check if any valid tags exist at position 2
        # Apply beam search - keep only top beam_size candidates
        if len(viterbi[2]) > beam_size:
            sorted_tags = sorted(viterbi[2].items(), key=lambda x: x[1], reverse=True)[:beam_size]
            viterbi[2] = {tag: score for tag, score in sorted_tags}
            bp[2] = {tag: bp[2][tag] for tag in viterbi[2]}

        # Normalize scores at position 2 (convert from score to probability)
        max_score = max(viterbi[2].values())
        normalizer = max_score + math.log(sum(math.exp(score - max_score) for score in viterbi[2].values()))
        for u in viterbi[2]:
            viterbi[2][u] -= normalizer  # Normalize in log space: log(p/sum) = log(p) - log(sum)

    # Fill the table for positions 3 to n-1
    for k in range(3, n):
        viterbi[k] = {}
        bp[k] = {}

        # For each possible current tag
        for u in tags:
            max_score = float('-inf')
            max_tag = None

            # For each possible previous tag (that is in our beam)
            for v in viterbi[k - 1]:
                # Early pruning - skip if previous probability is too low
                if viterbi[k - 1][v] < probability_threshold:
                    continue

                # Create history
                prev_prev_tag = bp[k - 1][v]  # Get tag from two positions back
                history = (sentence[k], None, sentence[k - 1], v, sentence[k - 2], prev_prev_tag,
                           sentence[k + 1] if k + 1 < n else "~")

                # Calculate score for this tag given history
                q_score = calc_q_prob(history, u, pre_trained_weights, feature2id)

                # Total score: previous score + transition score
                score = viterbi[k - 1][v] + q_score

                # Keep track of the best previous tag
                if score > max_score:
                    max_score = score
                    max_tag = v

            # Store best score and backpointer
            if max_tag is not None:  # Only store if we found a valid path
                viterbi[k][u] = max_score
                bp[k][u] = max_tag

        # Apply beam search - keep only top beam_size candidates
        if len(viterbi[k]) > beam_size:
            sorted_tags = sorted(viterbi[k].items(), key=lambda x: x[1], reverse=True)[:beam_size]
            viterbi[k] = {tag: score for tag, score in sorted_tags}
            bp[k] = {tag: bp[k][tag] for tag in viterbi[k]}

        # Normalize scores at position k if needed
        if viterbi[k]:  # Check if any valid tags exist
            max_score = max(viterbi[k].values())
            normalizer = max_score + math.log(sum(math.exp(score - max_score) for score in viterbi[k].values()))
            for u in viterbi[k]:
                viterbi[k][u] -= normalizer

    # Handle the last position (with special end symbol '~')
    if n - 1 not in viterbi:
        viterbi.append({})
        bp.append({})

    # Find the most likely tag for the last position
    if n - 1 > 2 and viterbi[n - 2]:  # Make sure we have valid previous states
        last_tag = max(viterbi[n - 2].items(), key=lambda x: x[1])[0]
    else:
        # Fallback if no valid path
        last_tag = tags[0] if tags else "NN"  # Default to NN if no tags

    # Build tag sequence by backtracking
    tag_sequence = ['~']  # Start with end symbol
    tag_sequence.append(last_tag)

    # Backtrack from the last position
    for k in range(n - 2, 2, -1):
        if tag_sequence[-1] in bp[k]:
            prev_tag = bp[k][tag_sequence[-1]]
            tag_sequence.append(prev_tag)
        else:
            # Fallback if no valid path
            tag_sequence.append(tags[0] if tags else "NN")

    # Add start symbols
    tag_sequence.append('*')
    tag_sequence.append('*')

    # Reverse to get the correct order
    tag_sequence.reverse()
    return tag_sequence


def simple_baseline_tagger(sentence, feature2id):
    """
    A simple baseline tagger that assigns the most common tag for each word
    Falls back to 'NN' for unknown words
    """
    word_to_tag = {}
    feature_stats = feature2id.feature_statistics

    # Find most common tag for each word in training data
    for word in feature_stats.words_count:
        max_count = 0
        best_tag = "NN"  # Default tag
        for tag in feature_stats.tags:
            if (word, tag) in feature_stats.feature_rep_dict["f100"]:
                count = feature_stats.feature_rep_dict["f100"][(word, tag)]
                if count > max_count:
                    max_count = count
                    best_tag = tag
        word_to_tag[word] = best_tag

    # Default tag is the most common overall
    default_tag = "NN"
    if feature_stats.tags_counts:
        default_tag = max(feature_stats.tags_counts.items(), key=lambda x: x[1])[0]

    # Tag the sentence
    tags = ['*', '*']  # Start symbols

    # Skip the first two words (which are actually '*' symbols) and the last one ('~')
    for i in range(2, len(sentence) - 1):
        word = sentence[i]
        tags.append(word_to_tag.get(word, default_tag))

    # Add end symbol
    tags.append('~')

    return tags


def batch_process_sentences(sentences, pre_trained_weights, feature2id, batch_size=10):
    """
    Process sentences in batches for better performance
    """
    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_results = []

        for sentence in batch:
            pred = memm_viterbi(sentence[0], pre_trained_weights, feature2id)

            # If Viterbi fails, fall back to baseline
            if not all(pred) or len(pred) < len(sentence[0]):
                print(f"Warning: Viterbi failed on a sentence, using baseline")
                pred = simple_baseline_tagger(sentence[0], feature2id)

            batch_results.append(pred)

        results.extend(batch_results)

    return results


def tag_all_test(test_path, pre_trained_weights, feature2id, predictions_path):
    """
    Tags all sentences in a test file and writes the output to a prediction file
    """
    tagged = "test" in test_path
    test = read_test(test_path, tagged=tagged)

    # Clear previous content if file exists
    with open(predictions_path, "w") as f:
        pass

    # Append to file
    output_file = open(predictions_path, "a+")

    # Process in batches for better performance
    batch_size = 5  # Adjust based on your hardware capabilities
    total_correct = 0
    total_words = 0

    for batch_start in tqdm(range(0, len(test), batch_size), desc="Processing sentences"):
        batch = test[batch_start:batch_start + batch_size]

        for k, sen in enumerate(batch):
            sentence = sen[0]
            gold_tags = sen[1] if tagged else None

            # Use the Viterbi algorithm for tagging
            pred = memm_viterbi(sentence, pre_trained_weights, feature2id)

            # If Viterbi fails, fall back to baseline
            if not all(pred) or len(pred) < len(sentence):
                print(f"Warning: Viterbi failed on sentence {batch_start + k}, using baseline")
                pred = simple_baseline_tagger(sentence, feature2id)

            # Skip the first two tags ('*', '*') and take only up to the last word (exclude '~')
            pred = pred[2:len(sentence)]
            sentence = sentence[2:len(sentence)]

            # Calculate accuracy if we have gold tags
            if tagged and gold_tags:
                gold_tags = gold_tags[2:len(sen[1])]
                for i in range(len(pred)):
                    if i < len(gold_tags):
                        total_words += 1
                        if pred[i] == gold_tags[i]:
                            total_correct += 1

            # Write tagged sentence to output file
            for i in range(len(sentence) - 1):  # Skip the last '~'
                if i > 0:
                    output_file.write(" ")
                output_file.write(f"{sentence[i]}_{pred[i]}")
            output_file.write("\n")

    output_file.close()

    # Return accuracy if tagged
    accuracy = total_correct / total_words if tagged and total_words > 0 else None
    if accuracy is not None:
        print(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_words} words correctly tagged)")

    return accuracy


def calculate_accuracy(true_tags_path, predicted_tags_path):
    """
    Calculate word-level accuracy of the model
    @param true_tags_path: Path to file with true tags
    @param predicted_tags_path: Path to file with predicted tags
    @return: Accuracy score
    """
    true_words_tags = []
    with open(true_tags_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        true_words_tags.append((parts[0], parts[1]))

    pred_words_tags = []
    with open(predicted_tags_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        pred_words_tags.append((parts[0], parts[1]))

    if len(true_words_tags) != len(pred_words_tags):
        print(f"Warning: length mismatch! True: {len(true_words_tags)}, Pred: {len(pred_words_tags)}")
        min_len = min(len(true_words_tags), len(pred_words_tags))
        true_words_tags = true_words_tags[:min_len]
        pred_words_tags = pred_words_tags[:min_len]

    correct = sum(1 for (t_word, t_tag), (p_word, p_tag) in zip(true_words_tags, pred_words_tags)
                  if t_word == p_word and t_tag == p_tag)
    total = len(true_words_tags)
    accuracy = correct / total if total > 0 else 0

    print(f"Accuracy: {accuracy:.4f} ({correct}/{total} words correctly tagged)")
    return accuracy


def create_confusion_matrix(true_tags_path, predicted_tags_path):
    """
    Create a confusion matrix to analyze tagging errors
    @param true_tags_path: Path to file with true tags
    @param predicted_tags_path: Path to file with predicted tags
    @return: Confusion matrix as a dictionary of dictionaries
    """
    true_tags = []
    pred_tags = []

    # Read true tags
    with open(true_tags_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        true_tags.append(parts[1])

    # Read predicted tags
    with open(predicted_tags_path, 'r') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        pred_tags.append(parts[1])

    # Ensure same length
    if len(true_tags) != len(pred_tags):
        min_len = min(len(true_tags), len(pred_tags))
        true_tags = true_tags[:min_len]
        pred_tags = pred_tags[:min_len]

    # Get unique tags
    all_tags = sorted(set(true_tags) | set(pred_tags))

    # Create confusion matrix
    confusion = {true_tag: {pred_tag: 0 for pred_tag in all_tags} for true_tag in all_tags}

    # Fill confusion matrix
    for t, p in zip(true_tags, pred_tags):
        confusion[t][p] += 1

    return confusion


def analyze_confusion_matrix(confusion_matrix, top_n=10):
    """
    Analyze and print results from confusion matrix
    """
    print(f"\n=== Confusion Matrix Analysis ===")

    # Extract errors from confusion matrix
    errors = []
    for true_tag, pred_dict in confusion_matrix.items():
        for pred_tag, count in pred_dict.items():
            if true_tag != pred_tag and count > 0:
                errors.append((true_tag, pred_tag, count))

    # Sort errors by count (descending)
    errors.sort(key=lambda x: x[2], reverse=True)

    # Calculate overall statistics
    total_errors = sum(count for _, _, count in errors)
    total_tags = sum(sum(pred_dict.values()) for pred_dict in confusion_matrix.values())
    accuracy = 1 - (total_errors / total_tags) if total_tags > 0 else 0

    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Total errors: {total_errors} out of {total_tags} tags")

    # Display top errors
    print(f"\nTop {top_n} most common tagging errors:")
    for i, (true_tag, pred_tag, count) in enumerate(errors[:top_n], 1):
        error_pct = (count / total_errors) * 100 if total_errors > 0 else 0
        print(f"{i}. True: {true_tag}, Predicted: {pred_tag}, Count: {count} ({error_pct:.2f}% of all errors)")

    # Calculate error rate by tag
    tag_totals = {}
    tag_errors = {}

    for true_tag, pred_dict in confusion_matrix.items():
        tag_total = sum(pred_dict.values())
        tag_totals[true_tag] = tag_total

        # Calculate errors for this tag
        tag_error = tag_total - pred_dict.get(true_tag, 0)
        tag_errors[true_tag] = tag_error

    # Sort tags by error rate
    tag_error_rates = [(tag, tag_errors[tag] / tag_totals[tag] * 100)
                       for tag in tag_totals if tag_totals[tag] > 0]
    tag_error_rates.sort(key=lambda x: x[1], reverse=True)

    # Display error rates for most problematic tags
    print(f"\nTop {top_n} most problematic tags:")
    print(f"{'Tag':<8} {'Error Rate':>10} {'Total':>8} {'Errors':>8}")
    print("-" * 36)

    for tag, error_rate in tag_error_rates[:top_n]:
        print(f"{tag:<8} {error_rate:>10.2f}% {tag_totals[tag]:>8} {tag_errors[tag]:>8}")

    return errors, tag_error_rates


def find_common_error_contexts(true_tags_path, predicted_tags_path, top_n=5):
    """
    Find common contextual patterns where errors occur
    """
    # Read the files
    true_sentences = []
    pred_sentences = []

    with open(true_tags_path, 'r') as file:
        for line in file:
            if line.strip():
                true_sentences.append(line.strip().split())

    with open(predicted_tags_path, 'r') as file:
        for line in file:
            if line.strip():
                pred_sentences.append(line.strip().split())

    # Ensure same number of sentences
    min_sentences = min(len(true_sentences), len(pred_sentences))
    true_sentences = true_sentences[:min_sentences]
    pred_sentences = pred_sentences[:min_sentences]

    # Find error contexts
    error_contexts = {}

    for true_sent, pred_sent in zip(true_sentences, pred_sentences):
        min_len = min(len(true_sent), len(pred_sent))

        for i in range(min_len):
            true_word_tag = true_sent[i].split('_')
            pred_word_tag = pred_sent[i].split('_')

            if len(true_word_tag) == 2 and len(pred_word_tag) == 2:
                true_word, true_tag = true_word_tag
                pred_word, pred_tag = pred_word_tag

                # Check if it's an error
                if true_tag != pred_tag:
                    # Get context (previous and next word)
                    prev_word = true_sent[i - 1].split('_')[0] if i > 0 else "START"
                    next_word = true_sent[i + 1].split('_')[0] if i < min_len - 1 else "END"

                    context = (prev_word, true_word, next_word)
                    error_type = (true_tag, pred_tag)

                    key = (context, error_type)
                    error_contexts[key] = error_contexts.get(key, 0) + 1

    # Find most common error contexts
    sorted_contexts = sorted(error_contexts.items(), key=lambda x: x[1], reverse=True)

    print("\n=== Common Error Contexts ===")
    for (context, error_type), count in sorted_contexts[:top_n]:
        prev, word, next_word = context
        true_tag, pred_tag = error_type
        print(f"Context: '{prev} [{word}] {next_word}' - True: {true_tag}, Pred: {pred_tag}, Count: {count}")

    return sorted_contexts
