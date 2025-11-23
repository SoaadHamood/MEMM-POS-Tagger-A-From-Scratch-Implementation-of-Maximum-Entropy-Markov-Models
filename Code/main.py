import pickle
import os
import argparse
import numpy as np
from collections import Counter, defaultdict
from preprocessing import preprocess_train, read_test
from optimization import get_optimal_vector
from inference import tag_all_test, calculate_accuracy, create_confusion_matrix


def analyze_dataset(file_path, is_tagged=True):
    """
    Analyze dataset statistics
    """
    print(f"\nAnalyzing dataset: {file_path}")

    if is_tagged:
        sentences = read_test(file_path, tagged=True)

        all_words = []
        all_tags = []
        sentence_lengths = []

        for sentence in sentences:
            words = sentence[0][2:-1]  # Skip start and end symbols
            tags = sentence[1][2:-1]  # Skip start and end symbols
            all_words.extend(words)
            all_tags.extend(tags)
            sentence_lengths.append(len(words))

        word_count = len(all_words)
        unique_words = len(set(all_words))
        tag_count = len(all_tags)
        unique_tags = len(set(all_tags))

        tag_freq = Counter(all_tags)

        print(f"Dataset contains {len(sentences)} sentences, {word_count} words")
        print(f"Vocabulary size: {unique_words}, Unique tags: {unique_tags}")
        print(f"Average sentence length: {sum(sentence_lengths) / len(sentence_lengths):.2f}")

        print("\nTop 5 most frequent tags:")
        for tag, count in tag_freq.most_common(5):
            print(f"  {tag}: {count} ({count / tag_count * 100:.2f}%)")

        return {
            'word_count': word_count,
            'unique_words': set(all_words),
            'unique_tags': set(all_tags),
            'tag_freq': tag_freq
        }
    else:
        sentences = read_test(file_path, tagged=False)
        all_words = []

        for sentence in sentences:
            words = sentence[0][2:-1]  # Skip start and end symbols
            all_words.extend(words)

        print(f"Dataset contains {len(sentences)} sentences, {len(all_words)} words")
        print(f"Vocabulary size: {len(set(all_words))}")

        return {
            'word_count': len(all_words),
            'unique_words': set(all_words)
        }


def analyze_errors(confusion_matrix, top_n=10):
    """
    Analyze common tagging errors
    """
    # Extract errors from confusion matrix
    errors = []
    for true_tag, pred_dict in confusion_matrix.items():
        for pred_tag, count in pred_dict.items():
            if true_tag != pred_tag and count > 0:
                errors.append((true_tag, pred_tag, count))

    # Sort errors by count (descending)
    errors.sort(key=lambda x: x[2], reverse=True)

    # Calculate error statistics
    total_errors = sum(count for _, _, count in errors)
    total_tags = sum(sum(pred_dict.values()) for pred_dict in confusion_matrix.values())

    print(f"\nError Analysis:")
    print(f"Total errors: {total_errors} out of {total_tags} tags")

    # Display top errors
    print(f"\nTop {top_n} most common tagging errors:")
    for i, (true_tag, pred_tag, count) in enumerate(errors[:top_n], 1):
        print(f"{i}. True: {true_tag}, Predicted: {pred_tag}, Count: {count}")

    return errors


def analyze_word_level_errors(true_tags_path, predicted_tags_path, top_n=20):
    """
    Analyze the most common word-level errors
    """
    # Read the files
    true_word_tags = []
    pred_word_tags = []

    with open(true_tags_path, 'r') as file:
        for line in file:
            if line.strip():
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        true_word_tags.append((parts[0], parts[1]))

    with open(predicted_tags_path, 'r') as file:
        for line in file:
            if line.strip():
                word_tags = line.strip().split()
                for word_tag in word_tags:
                    parts = word_tag.split('_')
                    if len(parts) == 2:
                        pred_word_tags.append((parts[0], parts[1]))

    # Ensure lengths match
    if len(true_word_tags) != len(pred_word_tags):
        min_len = min(len(true_word_tags), len(pred_word_tags))
        true_word_tags = true_word_tags[:min_len]
        pred_word_tags = pred_word_tags[:min_len]

    # Find errors
    word_errors = []
    for (true_word, true_tag), (pred_word, pred_tag) in zip(true_word_tags, pred_word_tags):
        if true_tag != pred_tag:
            word_errors.append((true_word, true_tag, pred_tag))

    # Count common word errors
    word_error_counts = Counter(word_errors)

    # Print most common word errors
    print(f"\nTop {top_n} most common mistagged words:")
    print(f"{'Word':<20} {'True Tag':<10} {'Predicted Tag':<15} {'Count':<10}")
    print("-" * 60)

    for (word, true_tag, pred_tag), count in word_error_counts.most_common(top_n):
        print(f"{word:<20} {true_tag:<10} {pred_tag:<15} {count:<10}")

    return word_error_counts


def analyze_error_contexts(true_tags_path, predicted_tags_path, top_n=10):
    """
    Analyze the context where errors occur
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

    # Ensure same length
    min_len = min(len(true_sentences), len(pred_sentences))
    true_sentences = true_sentences[:min_len]
    pred_sentences = pred_sentences[:min_len]

    # Analyze contexts
    contexts = []

    for true_sent, pred_sent in zip(true_sentences, pred_sentences):
        sent_len = min(len(true_sent), len(pred_sent))

        for i in range(sent_len):
            if "_" in true_sent[i] and "_" in pred_sent[i]:
                true_parts = true_sent[i].split('_')
                pred_parts = pred_sent[i].split('_')

                if len(true_parts) == 2 and len(pred_parts) == 2:
                    word, true_tag = true_parts
                    _, pred_tag = pred_parts

                    if true_tag != pred_tag:
                        # Get context (previous and next words if available)
                        prev_word = true_sent[i - 1].split('_')[0] if i > 0 else "START"
                        next_word = true_sent[i + 1].split('_')[0] if i < sent_len - 1 else "END"

                        context = (prev_word, word, next_word, true_tag, pred_tag)
                        contexts.append(context)

    # Count common contexts
    context_counts = Counter(contexts)

    # Print most common error contexts
    print(f"\nTop {top_n} most common error contexts:")
    print(f"{'Previous':<15} {'Word':<15} {'Next':<15} {'True Tag':<10} {'Pred Tag':<10} {'Count':<10}")
    print("-" * 80)

    for (prev, word, next_word, true_tag, pred_tag), count in context_counts.most_common(top_n):
        print(f"{prev:<15} {word:<15} {next_word:<15} {true_tag:<10} {pred_tag:<10} {count:<10}")

    return context_counts


def analyze_feature_weights(pre_trained_weights, feature2id, top_n=10):
    """
    Analyze the most influential features based on their weights
    """
    print(f"\nFeature Weight Analysis:")

    # Get feature weights by magnitude
    weight_features = []

    for feat_class in feature2id.feature_to_idx:
        for feat, idx in feature2id.feature_to_idx[feat_class].items():
            weight = pre_trained_weights[idx]
            weight_features.append((abs(weight), weight, feat_class, feat))

    # Sort by absolute weight (descending)
    weight_features.sort(reverse=True)

    # Display top features
    print(f"Top {top_n} most influential features:")
    for i, (_, weight, feat_class, feat) in enumerate(weight_features[:top_n], 1):
        feat_str = str(feat)
        if len(feat_str) > 50:
            feat_str = feat_str[:47] + "..."
        print(f"{i}. {feat_class}: {feat_str} (weight: {weight:.4f})")

    # Display average weight by feature class
    class_weights = defaultdict(list)
    for _, weight, feat_class, _ in weight_features:
        class_weights[feat_class].append(abs(weight))

    print("\nAverage absolute weight by feature class:")
    for feat_class, weights in sorted(class_weights.items(),
                                      key=lambda x: sum(x[1]) / len(x[1]),
                                      reverse=True):
        print(f"  {feat_class}: {sum(weights) / len(weights):.4f}")


def run_model1(threshold=20, lam=1.0):
    """
    Run Model 1 (Large Model)
    """
    print("\n" + "=" * 60)
    print("MODEL 1 (LARGE MODEL)".center(60))
    print("=" * 60)

    # Set paths
    train_path = "data/train1.wtag"
    test_path = "data/test1.wtag"
    comp_path = "data/comp1.words"

    weights_path = 'trained_models/weights_1.pkl'
    predictions_path = 'comp_m1_212794762.wtag'
    test_predictions_path = 'test_predictions_model1.wtag'

    # Create directory if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)

    # Analyze training data
    print("\nSTEP 1: Analyzing training data")
    train_stats = analyze_dataset(train_path)

    # Analyze test data if available
    if os.path.exists(test_path):
        print("\nSTEP 2: Analyzing test data")
        test_stats = analyze_dataset(test_path)

        # Compare vocabulary coverage
        train_vocab = train_stats['unique_words']
        test_vocab = test_stats['unique_words']
        oov_words = test_vocab - train_vocab

        print(f"\nVocabulary coverage:")
        print(f"  Words in test not seen in training: {len(oov_words)} "
              f"({len(oov_words) / len(test_vocab) * 100:.2f}% of test vocabulary)")

    # Train the model
    print("\nSTEP 3: Training model")
    print(f"Using threshold={threshold} and lambda={lam}")

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    print(f"Model saved to {weights_path}")

    # Load the model
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    # Evaluate on test data
    if os.path.exists(test_path):
        print("\nSTEP 4: Evaluating model on test data")

        # --------------- START OF MODIFIED SECTION ---------------
        # Tag test data and capture the accuracy value
        # but we'll ignore this accuracy and calculate it again
        # using our consistent method
        _ = tag_all_test(test_path, pre_trained_weights, feature2id, test_predictions_path)

        # Add a separator to make output clearer
        print("\n" + "-" * 50)
        print("OFFICIAL ACCURACY CALCULATION".center(50))
        print("-" * 50)

        # Calculate accuracy separately for consistency
        accuracy = calculate_accuracy(test_path, test_predictions_path)

        print(f"\nFINAL MODEL 1 ACCURACY: {accuracy:.4f}")
        print("-" * 50 + "\n")
        # --------------- END OF MODIFIED SECTION ---------------

        # ENHANCED ERROR ANALYSIS
        print("\nSTEP 5: Performing enhanced error analysis")

        # Tag-level confusion matrix
        confusion_matrix = create_confusion_matrix(test_path, test_predictions_path)
        analyze_errors(confusion_matrix)

        # Word-level error analysis (NEW)
        print("\nDetailed Word-Level Error Analysis:")
        word_errors = analyze_word_level_errors(test_path, test_predictions_path)

        # Error context analysis (NEW)
        print("\nError Context Analysis:")
        analyze_error_contexts(test_path, test_predictions_path)

        # Feature weight analysis
        analyze_feature_weights(pre_trained_weights, feature2id)

    # Generate competition predictions
    print("\nSTEP 6: Generating competition predictions")
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_path)
    print(f"Competition predictions saved to {predictions_path}")

    return weights_path, feature2id


def run_model2(threshold=20, lam=0.8):
    """
    Run Model 2 (Small Model)
    """
    print("\n" + "=" * 60)
    print("MODEL 2 (SMALL MODEL)".center(60))
    print("=" * 60)

    # Set paths
    train_path = "data/train2.wtag"
    comp_path = "data/comp2.words"

    weights_path = 'trained_models/weights_2.pkl'
    predictions_path = 'comp_m2_212794762.wtag'

    # Create directory if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)

    # Analyze training data
    print("\nSTEP 1: Analyzing training data")
    train_stats = analyze_dataset(train_path)

    # Train the model
    print("\nSTEP 2: Training model")
    print(f"Using threshold={threshold} and lambda={lam}")

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    print(f"Model saved to {weights_path}")

    # Load the model
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    # Since we don't have test data for Model 2, estimate performance with cross-validation
    print("\nSTEP 3: Estimating model performance")
    print("Note: No separate test set available for Model 2")
    print("      Expected accuracy based on similar small-data POS tagging models: ~85-90%")

    # Feature weight analysis
    print("\nSTEP 4: Analyzing feature weights")
    analyze_feature_weights(pre_trained_weights, feature2id)

    # Generate competition predictions
    print("\nSTEP 5: Generating competition predictions")
    tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_path)
    print(f"Competition predictions saved to {predictions_path}")

    return weights_path, feature2id


def cross_validate_model2(train_path, k=5, threshold=3, lam=0.8):
    """
    Perform k-fold cross-validation on Model 2 dataset
    """
    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION FOR MODEL 2 (K={k})".center(60))
    print(f"{'=' * 60}")

    # Read all sentences from the training file
    print(f"\nReading data from {train_path}")
    with open(train_path, 'r') as f:
        all_sentences = [line.strip() for line in f if line.strip()]

    print(f"Total sentences: {len(all_sentences)}")

    # Set up k-fold cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Temporary file paths
    tmp_train_path = "temp_train.wtag"
    tmp_test_path = "temp_test.wtag"
    tmp_weights_path = "temp_weights.pkl"
    tmp_predictions_path = "temp_predictions.wtag"

    # Store accuracy for each fold
    accuracies = []
    all_true_tags = []
    all_pred_tags = []

    # Run cross-validation
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_sentences)):
        print(f"\n{'=' * 40}")
        print(f"FOLD {fold + 1}/{k}".center(40))
        print(f"{'=' * 40}")

        # Create temporary train file
        with open(tmp_train_path, 'w') as f:
            for idx in train_idx:
                f.write(all_sentences[idx] + '\n')

        # Create temporary test file
        with open(tmp_test_path, 'w') as f:
            for idx in test_idx:
                f.write(all_sentences[idx] + '\n')

        # Train model on this fold
        print(f"\nTraining model for fold {fold + 1}")
        statistics, feature2id = preprocess_train(tmp_train_path, threshold)
        get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=tmp_weights_path, lam=lam)

        # Load trained model
        with open(tmp_weights_path, 'rb') as f:
            optimal_params, feature2id = pickle.load(f)
        pre_trained_weights = optimal_params[0]

        # Evaluate on test fold
        print(f"\nEvaluating model on fold {fold + 1}")
        # Tag test data but ignore the return value (the first accuracy calculation)
        _ = tag_all_test(tmp_test_path, pre_trained_weights, feature2id, tmp_predictions_path)

        # Add a separator for clarity
        print("\n" + "-" * 40)
        print("OFFICIAL ACCURACY FOR FOLD".center(40))
        print("-" * 40)

        # Calculate accuracy consistently
        accuracy = calculate_accuracy(tmp_test_path, tmp_predictions_path)
        print(f"Fold {fold + 1} final accuracy: {accuracy:.4f}")
        print("-" * 40)

        accuracies.append(accuracy)

        # Collect true and predicted tags for overall analysis
        with open(tmp_test_path, 'r') as f:
            test_lines = [line.strip() for line in f if line.strip()]

        with open(tmp_predictions_path, 'r') as f:
            pred_lines = [line.strip() for line in f if line.strip()]

        for test_line, pred_line in zip(test_lines, pred_lines):
            # Extract tags from test line
            test_tokens = test_line.split()
            test_tags = [token.split('_')[1] for token in test_tokens]

            # Extract tags from prediction line
            pred_tokens = pred_line.split()
            pred_tags = [token.split('_')[1] for token in pred_tokens]

            # Ensure same length
            min_len = min(len(test_tags), len(pred_tags))
            all_true_tags.extend(test_tags[:min_len])
            all_pred_tags.extend(pred_tags[:min_len])

    # Calculate average accuracy
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    print(f"\n{'=' * 60}")
    print(f"CROSS-VALIDATION RESULTS".center(60))
    print(f"{'=' * 60}")

    print(f"\nFold accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"Fold {i + 1}: {acc:.4f}")

    print(f"\nOverall accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"95% confidence interval: [{mean_accuracy - 1.96 * std_accuracy / np.sqrt(k):.4f}, "
          f"{mean_accuracy + 1.96 * std_accuracy / np.sqrt(k):.4f}]")

    # Analyze errors by tag
    tag_counts = Counter(all_true_tags)
    tag_errors = defaultdict(int)

    for true_tag, pred_tag in zip(all_true_tags, all_pred_tags):
        if true_tag != pred_tag:
            tag_errors[true_tag] += 1

    print("\nError analysis by tag:")
    print(f"{'Tag':<8} {'Error Rate':>10} {'Total':>8} {'Errors':>8}")
    print("-" * 40)

    # Sort by error rate
    for tag in sorted(tag_counts.keys(),
                      key=lambda t: tag_errors[t] / tag_counts[t] if tag_counts[t] > 0 else 0,
                      reverse=True):
        count = tag_counts[tag]
        errors = tag_errors[tag]
        error_rate = errors / count * 100 if count > 0 else 0

        if count >= 10:  # Only show tags with reasonable frequency
            print(f"{tag:<8} {error_rate:>10.2f}% {count:>8} {errors:>8}")

    # Clean up temporary files
    for tmp_file in [tmp_train_path, tmp_test_path, tmp_weights_path, tmp_predictions_path]:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

    return mean_accuracy, std_accuracy


def run_model2(threshold=3, lam=0.8):
    """
    Run Model 2 (Small Model)
    """
    print("\n" + "=" * 60)
    print("MODEL 2 (SMALL MODEL)".center(60))
    print("=" * 60)

    # Set paths
    train_path = "data/train2.wtag"
    comp_path = "data/comp2.words"

    weights_path = 'trained_models/weights_2.pkl'
    predictions_path = 'comp_m2_212794762.wtag'

    # Create directory if it doesn't exist
    os.makedirs('trained_models', exist_ok=True)

    # Analyze training data
    print("\nSTEP 1: Analyzing training data")
    train_stats = analyze_dataset(train_path)

    # Train the model
    print("\nSTEP 2: Training model")
    print(f"Using threshold={threshold} and lambda={lam}")

    statistics, feature2id = preprocess_train(train_path, threshold)
    get_optimal_vector(statistics=statistics, feature2id=feature2id, weights_path=weights_path, lam=lam)

    print(f"Model saved to {weights_path}")

    # Load the model
    with open(weights_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    # Since we don't have test data for Model 2, estimate performance with cross-validation
    print("\nSTEP 3: Estimating model performance with cross-validation")

    # Call the cross-validation function
    mean_accuracy, std_accuracy = cross_validate_model2(train_path, k=5, threshold=threshold, lam=lam)

    print(f"\nCross-validation results summary:")
    print(f"Final Model 2 accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")

    # Feature weight analysis
    print("\nSTEP 4: Analyzing feature weights")
    analyze_feature_weights(pre_trained_weights, feature2id)

    # Generate competition predictions
    print("\nSTEP 5: Generating competition predictions")

    # Capture and ignore the accuracy value returned by tag_all_test
    _ = tag_all_test(comp_path, pre_trained_weights, feature2id, predictions_path)

    print(f"Competition predictions saved to {predictions_path}")

    return weights_path, feature2id
def generate_comp_tagged():
    """
    Generate competition tagged files using existing models
    """
    print("\n" + "=" * 60)
    print("GENERATING COMPETITION FILES".center(60))
    print("=" * 60)

    # Paths
    model1_path = 'trained_models/weights_1.pkl'
    model2_path = 'trained_models/weights_2.pkl'
    comp1_path = "data/comp1.words"
    comp2_path = "data/comp2.words"

    # Check if models exist
    if not os.path.exists(model1_path):
        print(f"Error: Model 1 file not found at {model1_path}")
        print("Please train Model 1 first")
        return

    if not os.path.exists(model2_path):
        print(f"Error: Model 2 file not found at {model2_path}")
        print("Please train Model 2 first")
        return

    # Load Model 1 and generate predictions
    print("\nGenerating predictions for Model 1...")
    with open(model1_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    comp1_out = "comp_m1_212794762.wtag"
    tag_all_test(comp1_path, pre_trained_weights, feature2id, comp1_out)
    print(f"Model 1 predictions saved to {comp1_out}")

    # Load Model 2 and generate predictions
    print("\nGenerating predictions for Model 2...")
    with open(model2_path, 'rb') as f:
        optimal_params, feature2id = pickle.load(f)
    pre_trained_weights = optimal_params[0]

    comp2_out = "comp_m2_212794762.wtag"
    tag_all_test(comp2_path, pre_trained_weights, feature2id, comp2_out)
    print(f"Model 2 predictions saved to {comp2_out}")

    print("\nCompetition files generated successfully.")


def main():
    """
    Main function to parse arguments and run models
    """
    parser = argparse.ArgumentParser(description='MEMM Part-of-Speech Tagger')
    parser.add_argument('--mode', type=str, choices=['all', 'model1', 'model2', 'generate_comp'],
                        default='all', help='Which mode to run')
    parser.add_argument('--threshold1', type=int, default=20,
                        help='Feature count threshold for Model 1')
    parser.add_argument('--lambda1', type=float, dest='lam1', default=1.0,
                        help='Regularization parameter for Model 1')
    parser.add_argument('--threshold2', type=int, default=3,
                        help='Feature count threshold for Model 2')
    parser.add_argument('--lambda2', type=float, dest='lam2', default=0.8,
                        help='Regularization parameter for Model 2')

    args = parser.parse_args()

    if args.mode in ['all', 'model1']:
        run_model1(threshold=args.threshold1, lam=args.lam1)

    if args.mode in ['all', 'model2']:
        run_model2(threshold=args.threshold2, lam=args.lam2)

    if args.mode == 'generate_comp':
        generate_comp_tagged()


if __name__ == '__main__':
    main()