import json
import os.path as osp
import argparse
from collections import defaultdict
from datetime import datetime

# StreamingBench task-type labels (CSV)
TASK_TYPES = [
    'Object Perception', 
    'Causal Reasoning', 
    'Clips Summarize', 
    'Attribute Perception', 
    'Event Understanding', 
    'Text-Rich Understanding', 
    'Prospective Reasoning', 
    'Spatial Understanding', 
    'Action Perception', 
    'Counting'
]


def load_results(jsonl_path):
    """Load per-row JSON objects from a StreamingBench results JSONL."""
    results = []
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(f"results file not found: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Skip trailing aggregate row if present
                if 'accuracy' in item and 'task_type_accuracies' in item:
                    continue
                # Require fields needed for scoring
                if 'question_id' in item and 'task_type' in item and 'answer' in item and 'response' in item:
                    results.append(item)
            except json.JSONDecodeError as e:
                print(f"warning: skip invalid JSON line: {e}")
                continue
    
    return results


def evaluate_results(results):
    """Compute per-type and overall accuracy."""
    correct_tasks = {task_type: 0 for task_type in TASK_TYPES}
    total_tasks = {task_type: 0 for task_type in TASK_TYPES}
    
    # Per-task-type correct / total
    for item in results:
        task_type = item.get('task_type', 'Unknown')
        answer = item.get('answer', '').strip()
        response = item.get('response', '').strip()
        
        # Unknown task_type -> skip with notice
        if task_type not in TASK_TYPES:
            print(f"warning: unknown task_type={task_type}, question_id={item.get('question_id', 'unknown')}")
            continue
        
        total_tasks[task_type] += 1
        
        # Exact string match on model output
        if response == answer:
            correct_tasks[task_type] += 1
    
    # Per-type rates
    task_type_accuracies = {}
    for task_type in TASK_TYPES:
        if total_tasks[task_type] > 0:
            task_type_accuracies[task_type] = correct_tasks[task_type] / total_tasks[task_type]
        else:
            task_type_accuracies[task_type] = 0.0
    
    # Macro overall
    total_correct = sum(correct_tasks.values())
    total_all = sum(total_tasks.values())
    overall_accuracy = total_correct / total_all if total_all > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        "task_type_accuracies": task_type_accuracies,
        "correct_by_task": correct_tasks,
        "total_by_task": total_tasks,
        "total_correct": total_correct,
        "total_all": total_all
    }


def print_evaluation_results(eval_results):
    """Pretty-print tables to stdout."""
    print("=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    print(f"\nOverall accuracy: {eval_results['accuracy'] * 100:.2f}% ({eval_results['total_correct']}/{eval_results['total_all']})")
    print("\nPer task type:")
    print("-" * 80)
    
    for task_type in TASK_TYPES:
        correct = eval_results['correct_by_task'][task_type]
        total = eval_results['total_by_task'][task_type]
        accuracy = eval_results['task_type_accuracies'][task_type]
        
        if total > 0:
            print(f"  {task_type:30s}: {accuracy * 100:6.2f}% ({correct:4d}/{total:4d})")
        else:
            print(f"  {task_type:30s}: {'N/A':>6s} ({correct:4d}/{total:4d})")
    
    print("=" * 80)


def save_evaluation_results(eval_results, output_path):
    """Persist metrics JSON for downstream plotting."""
    output_dict = {
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": eval_results['accuracy'],
        "task_type_accuracies": eval_results['task_type_accuracies'],
        "correct_by_task": eval_results['correct_by_task'],
        "total_by_task": eval_results['total_by_task'],
        "total_correct": eval_results['total_correct'],
        "total_all": eval_results['total_all']
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\nWrote metrics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Score StreamingBench JSONL outputs")
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to model output JSONL",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to write metrics JSON"
    )
    args = parser.parse_args()
    
    print(f"Loading results: {args.result_file}")
    results = load_results(args.result_file)
    print(f"Loaded {len(results)} rows")
    
    print("\nScoring...")
    eval_results = evaluate_results(results)
    
    print_evaluation_results(eval_results)
    
    if args.output_file:
        save_evaluation_results(eval_results, args.output_file)


if __name__ == "__main__":
    main()

