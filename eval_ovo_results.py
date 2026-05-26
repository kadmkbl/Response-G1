import argparse
import json
import logging
import os.path as osp
from collections import defaultdict

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# OVO-Bench task families
backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

def score(results):
    def calculate_score_backward_realtime(results):
        def get_score(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        # Calculate Score for Every Result
        for i in range(len(results)):
            results[i]["score"] = get_score(results[i]["response"], results[i]["ground_truth"])
        
        scores = {}
        for i in range(len(results)):
            if not results[i]["task"] in scores.keys():
                scores[results[i]["task"]] = [results[i]["score"]]
            else:
                scores[results[i]["task"]].append(results[i]["score"])
        return results, scores

    def calculate_score_forward(results):
        def get_score_REC(response, gt):
            if response == None:
                return 0
            import re
            response = re.findall(r'\d+', response)
            response = "".join(response)
            return response == str(gt)
        
        def get_score_SSR_CRR(response, gt):
            if response == None:
                return 0
            return int(gt in response)
        
        scores = {}
        tasks = list(set([result["task"] for result in results]))
        for task in tasks:
            scores[task] = []
        for i, result in enumerate(results):
            # Calculate score for REC
            if result["task"] == "REC":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    cnt_correct += get_score_REC(test_info_["response"], test_info_["count"])
                scores["REC"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for SSR
            if result["task"] == "SSR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    response = test_info_["response"]
                    if (response == "N" and test_info_["type"] == 0) or (response == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(response, gt)
                scores["SSR"].append(cnt_correct / len(result["test_info"]))
            # Calculate score for CRR
            if result["task"] == "CRR":
                cnt_correct = 0
                for j, test_info_ in enumerate(result["test_info"]):
                    if (test_info_["response"] == "N" and test_info_["type"] == 0) or (test_info_["response"] == "Y" and test_info_["type"] == 1):
                        cnt_correct += 1
                        continue
                    gt = "No" if test_info_["type"] == 0 else "Yes"
                    cnt_correct += get_score_SSR_CRR(test_info_["response"], gt)
                scores["CRR"].append(cnt_correct / len(result["test_info"]))
        return results, scores
    
    backward_results = results.get("backward", [])
    realtime_results = results.get("realtime", [])
    forward_results = results.get("forward", [])
    avg_scores = {
        "backward": [],
        "realtime": [],
        "forward": []
    }

    if len(backward_results) > 0:
        # print("Evaluate Backward Tracing...")
        backward_results, backward_scores = calculate_score_backward_realtime(backward_results)
        # correct_backward, total_backward = 0, 0
        for k, v in backward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_backward += sum(v)
            # total_backward += len(v)
            avg_scores["backward"].append(sum(v)/len(v))
        # print(f"Backward Avg.: {100 * correct_backward / total_backward:.2f}\n")
        logger.info(f"Backward Avg.: {100 * sum(avg_scores['backward'])/len(avg_scores['backward']):.2f}\n")
    else:
        # correct_backward = 0
        # total_backward = 0
        pass
        
    if len(realtime_results) > 0:
        # print("Evaluate Real-time Visual Perception...")
        realtime_results, realtime_scores = calculate_score_backward_realtime(realtime_results)
        # correct_realtime, total_realtime = 0, 0
        for k, v in realtime_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_realtime += sum(v)
            # total_realtime += len(v)
            avg_scores["realtime"].append(sum(v)/len(v))
        # print(f"Realtime Avg.: {100 * correct_realtime / total_realtime:.2f}\n")
        logger.info(f"Realtime Avg.: {100 * sum(avg_scores['realtime'])/len(avg_scores['realtime']):.2f}\n")
    else:
        # correct_realtime = 0
        # total_realtime = 0
        pass

    if len(forward_results) > 0:
        # print("Evaluate Forward Active Responding...")
        forward_results, forward_scores = calculate_score_forward(forward_results)
        # correct_forward, total_forward = 0, 0
        for k, v in forward_scores.items():
            logger.info(f"Task: {k}, Acc: {100 * sum(v)/len(v):.2f}")
            # correct_forward += sum(v)
            # total_forward += len(v)
            avg_scores["forward"].append(sum(v)/len(v))
        # print(f"Forward Avg.: {100 * correct_forward / total_forward:.2f}\n")
        logger.info(f"Forward Avg.: {100 * sum(avg_scores['forward'])/len(avg_scores['forward']):.2f}\n")
    else:
        # correct_forward = 0
        # total_forward = 0
        pass

    # Macro average over non-empty splits
    total_avg = 0
    count = 0
    if len(avg_scores['backward']) > 0:
        total_avg += sum(avg_scores['backward'])/len(avg_scores['backward'])
        count += 1
    if len(avg_scores['realtime']) > 0:
        total_avg += sum(avg_scores['realtime'])/len(avg_scores['realtime'])
        count += 1
    if len(avg_scores['forward']) > 0:
        total_avg += sum(avg_scores['forward'])/len(avg_scores['forward'])
        count += 1
    
    if count > 0:
        logger.info(f"Total Avg.: {100 * total_avg / count:.2f}")
    else:
        logger.info("Total Avg.: No results to calculate")


def load_ovo_jsonl(jsonl_path):
    """Load OVO-Bench result rows from JSONL; bucket by task family."""
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(f"results file not found: {jsonl_path}")

    results = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("skip invalid JSON line: %s", e)
                continue
            task = item.get("task")
            if task in backward_tasks:
                results["backward"].append(item)
            elif task in realtime_tasks:
                results["realtime"].append(item)
            elif task in forward_tasks:
                results["forward"].append(item)
            else:
                logger.warning("skip row with unknown task=%r", task)
    return results


def main():
    parser = argparse.ArgumentParser(description="Score OVO-Bench JSONL outputs")
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to model output JSONL (OVO-Bench format)",
    )
    args = parser.parse_args()

    logger.info("Loading results: %s", args.result_file)
    results = load_ovo_jsonl(args.result_file)
    n = sum(len(v) for v in results.values())
    logger.info("Loaded %d rows into backward/realtime/forward buckets", n)
    score(results)


if __name__ == "__main__":
    main()
