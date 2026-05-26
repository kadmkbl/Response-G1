from transformers import AutoProcessor
import torch
import json
import os
import os.path as osp
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import ffmpeg
from collections import deque
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
from transformers import Qwen3VLForConditionalGeneration
import torch.nn.functional as F

from qwen_online.utils import encode_images, whole_current_frame_tokens
from response_graph.Scene_Graph import Query_Graph_generation, Scene_Graph_generation_po_frame, format_scene_graphs_for_prompt_offline, format_scene_graphs_for_prompt_query_offline

# Process / CUDA environment (set here or export before launch).
os.environ['NPROC_PER_NODE'] = '1'
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path and hyperparameter defaults are defined in build_parser() / resolved on args in main().

prompt = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

Question: {question}

Options:
{options}

The best option is:"""

prompt_graph_context = """You are an advanced video question-answering AI assistant. You have been provided with some frames from the video and a multiple-choice question related to the video. Your task is to carefully analyze the video and provide the best answer to question, choosing from the four options provided. Respond with only the letter (A, B, C, or D) of the correct option.

The relative scene graph shows:
{context_Scene_Graphs_text}

Question: {question}

Options:
{options}

The best option is:"""

text_kwargs = {
    "padding": True,
    "return_token_type_ids": False
}

generate_trigger_kwargs = {
    "do_sample": False,
    "num_beams": 1, 
    "min_length": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 128,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


class Memory_Bank_Naive:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_bank = deque(maxlen=memory_size)  # Rolling frame token bank (maxlen in frames).
        self.Scene_Graph_memory_bank = deque(maxlen=memory_size)  # Rolling (timestamp, graph) bank.
        print(f"Memory bank ready (max {memory_size} frames).")

    def update(self, current_frame):
        self.memory_bank.append(current_frame)  # FIFO via deque maxlen.

    def update_graph(self, timestamp, current_Scene_Graph):
        self.Scene_Graph_memory_bank.append((timestamp, current_Scene_Graph))  # FIFO via deque maxlen.

    def get_context_frames(self, context_window):
        start_index = max(0, len(self.memory_bank) - context_window)
        recent_frames = list(self.memory_bank)[start_index:]

        return recent_frames

    def get_newest_graph(self):
        if len(self.Scene_Graph_memory_bank) == 0:
            return None

        context_Scene_Graphs_list = list(self.Scene_Graph_memory_bank)
        newest_indices = [len(context_Scene_Graphs_list) - 1]  # Index of the newest graph.
        newest_Scene_Graphs = [context_Scene_Graphs_list[i] for i in newest_indices]

        return newest_Scene_Graphs


def build_prompt(question, options, context_Scene_Graphs):
    if context_Scene_Graphs is not None:
        context_Scene_Graphs_text = "The relative scene graph shows:\n"
        for timestamp, Scene_Graph in context_Scene_Graphs:
            frame_second_placeholder = f"<{timestamp:.1f} seconds>"
            context_Scene_Graph_text = format_scene_graphs_for_prompt_offline([Scene_Graph])
            context_Scene_Graphs_text += f"{frame_second_placeholder}:{context_Scene_Graph_text}\n"
        prompt_response = prompt_graph_context.format(question=question, options='\n'.join(eval(options)), context_Scene_Graphs_text=context_Scene_Graphs_text)
    else:
        prompt_response = prompt.format(question=question, options='\n'.join(eval(options)))

    return prompt_response


def eval_r(args):
    # Load Qwen3-VL weights and processor.
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.ckpt_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    logger.info(f"Load model and processor from {args.ckpt_path}")
    model.eval()

    # Load task annotations.
    task_df = pd.read_csv(args.task_csv)
    required_cols = [
        "question_id", "task_type", "question", "time_stamp", 
        "options", "answer", 
        "temporal_clue_type", "frames_required"
    ]
    for col in required_cols:
        if col not in task_df.columns:
            raise ValueError(f"Task CSV missing required column: {col} (check CSV format)")

    sg_generation_interval = args.sg_generation_interval
    sg_force_first_frame = args.sg_force_first_frame
    task_types = ['Object Perception', 'Causal Reasoning', 'Clips Summarize', 'Attribute Perception', 'Event Understanding', 'Text-Rich Understanding', 'Prospective Reasoning', 'Spatial Understanding', 'Action Perception', 'Counting']
    correct_tasks = {task_type: 0 for task_type in task_types}
    total_tasks = {task_type: 0 for task_type in task_types}  # Per-type denominators
    # Main evaluation loop.
    with torch.no_grad():
        for row in tqdm(task_df.itertuples(), total=len(task_df), desc="Processing Proactive Tasks"):
        # for row in tqdm(task_df_reverse.itertuples(), total=len(task_df_reverse), desc="Processing Proactive Tasks"):
            try:
                # Unpack one task row.
                question_id = row.question_id
                task_type = row.task_type
                total_tasks[task_type] += 1  # Per-type count
                question = row.question  # Natural-language trigger / question text.
                time_stamp = row.time_stamp  # Monitoring window start (e.g. HH:MM:SS).
                options = row.options
                answer = row.answer
                temporal_clue_type = row.temporal_clue_type  # Temporal clue tag from benchmark.
                frames_required = row.frames_required  # Frame budget hint from benchmark.

                # Build query-conditioned graph template from the question.
                Query_Graph = Query_Graph_generation(processor, model, question)
                # continue

                # Resolve video path from question_id.
                try:
                    video_idx = question_id.split('_')[-2]  # Convention: sample index is second-to-last token.
                except IndexError:
                    logger.warning(f"Invalid question_id format: {question_id}, skip this task")
                    continue
                video_path = osp.join(args.video_dir, f"sample_{video_idx}", "video.mp4")
                if not osp.exists(video_path):
                    logger.warning(f"Video file not found: {video_path}, skip this task")
                    continue

                # Temporal crop in frame indices.
                # tolerance_frame = args.tolerance_time * args.fps
                start_frame = 0
                end_frame = time_to_seconds(time_stamp) * args.fps
                max_frame = end_frame
                if 300 < end_frame <= 600:
                    args.frame_interval = 2
                elif end_frame > 600:
                    args.frame_interval = 5

                # Allocate streaming memory for this clip.
                memories = Memory_Bank_Naive(memory_size = int(max_frame / args.frame_interval) + 1)

                # Streaming Pipeline
                for frame_idx in range(start_frame, max_frame + 1, args.frame_interval):
                    timestamp = frame_idx / args.fps

                    current_frame = extract_frame(video_path, timestamp)
                    if current_frame == None:
                        continue
                    current_frame_embeddings = processor.image_processor(current_frame)
                    current_frame_embeddings["pixel_values"] = current_frame_embeddings["pixel_values"].to(model.device)
                    current_frame_embeddings['image_grid_thw'] = current_frame_embeddings['image_grid_thw'].to(model.device)
                    current_frame_tokens, _, token_per_frame = encode_images(model, current_frame_embeddings["pixel_values"], current_frame_embeddings['image_grid_thw'])
                    current_frame_tokens = whole_current_frame_tokens(processor, model, timestamp, current_frame_tokens)
                    current_frame_tokens = current_frame_tokens.unsqueeze(0).to(model.device)
                    # Drop vision tensors for the current frame to limit VRAM.
                    del current_frame_embeddings
                    if current_frame_tokens is None:
                        continue
                    else:
                        memories.update(current_frame_tokens)

                current_clip_tokens = memories.get_context_frames(context_window=1)
                current_clip_tokens = torch.cat(current_clip_tokens, dim=1).to(model.device)
                current_Scene_Graph = Scene_Graph_generation_po_frame(processor, model, current_clip_tokens, question)
                if len(current_Scene_Graph.get('scene_graph', [])) != 0:
                    memories.update_graph(timestamp, current_Scene_Graph)

                #* Retrieve Context_Scene_Graphs based on the Query_Graph *#
                context_Scene_Graphs = memories.get_newest_graph()
                # context_Scene_Graphs = None

                #* Build Prompt for Triggering and Response *#
                prompt_response = build_prompt(question, options, context_Scene_Graphs)
                response_ids = processor.tokenizer(
                    ["<|im_start|>user\n" + prompt_response +"<|im_end|>\n"], **text_kwargs)
                response_ids_tensor = torch.tensor(response_ids['input_ids']).to(model.device)
                response_prompt_embeddings = model.get_input_embeddings()(response_ids_tensor)

                past_frame_tokens = memories.get_context_frames(context_window=int(frame_idx/args.frame_interval)+1)
                past_frame_tokens = torch.cat(past_frame_tokens, dim=1).to(model.device)

                input_embeddings = torch.cat([past_frame_tokens, response_prompt_embeddings], dim=1).to(model.device)

                generation_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
                generation_ids_tensor = torch.tensor(generation_ids['input_ids']).to(model.device)
                generation_prompt_embeddings = model.get_input_embeddings()(generation_ids_tensor)
                input_embeddings = torch.cat([input_embeddings, generation_prompt_embeddings], dim=1).to(model.device)

                attention_masks = torch.ones(input_embeddings.shape[:2], dtype=torch.long).to(model.device)

                generated_ids = model.generate(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_masks,
                    **generate_trigger_kwargs,
                )
                output_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = output_text[0]

                # Release CUDA temporaries
                del input_embeddings, attention_masks, generated_ids, past_frame_tokens
                del response_prompt_embeddings, generation_prompt_embeddings
                del response_ids_tensor, generation_ids_tensor
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if response == answer:
                    correct_tasks[task_type] += 1

                # Append one JSON line per task.
                output_dict = {
                    'question_id': question_id,
                    'task_type': task_type,
                    'question': question,
                    'time_stamp': time_stamp,
                    'answer': answer,
                    'options': eval(options),
                    'frames_required': frames_required,
                    'temporal_clue_type': temporal_clue_type,
                    'response': response
                }

                with open(args.output_jsonl, 'a' if osp.exists(args.output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

            except Exception as e:
                logger.error(f"Error processing task {row.question_id if hasattr(row, 'question_id') else 'unknown'}: {str(e)}")
                print(f"Error processing task {row.question_id if hasattr(row, 'question_id') else 'unknown'}: {str(e)}")
                continue  # Skip bad rows and continue

    # Write aggregate metrics
    total_correct = sum(correct_tasks.values())
    total_all = len(task_df)
    task_type_accuracies = {}
    for task_type in task_types:
        if total_tasks[task_type] > 0:
            task_type_accuracies[task_type] = correct_tasks[task_type] / total_tasks[task_type]
        else:
            task_type_accuracies[task_type] = 0

    overall_results = {
        "accuracy": total_correct / total_all if total_all > 0 else 0,
        "task_type_accuracies": task_type_accuracies,
        "correct_by_task": correct_tasks,
        "total_by_task": total_tasks
    }
    with open(args.output_jsonl, 'a' if osp.exists(args.output_jsonl) else 'w') as f:
        f.write(json.dumps(overall_results, ensure_ascii=False) + '\n')


# helper functions
def time_to_seconds(time_str):
    """Parse HH:MM:SS or MM:SS wall-clock string to seconds."""
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def seconds_to_time(total_seconds):
    """Format integer seconds as zero-padded HH:MM:SS."""
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def extract_frame(video_path, timestamp):
    """Sample one RGB PIL frame at `timestamp` seconds (video-native FPS)."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS)  # Use container FPS, not args.fps
        frame_number = int(timestamp * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # BGR (OpenCV) -> RGB for PIL / processor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # import numpy as np
        # image_array = np.array(pil_image)
        # print(f"debug array shape: {image_array.shape}")

        return pil_image

    except Exception as e:
        print(f"extract_frame failed: {e}")
        return None


def build_parser():
    parser = argparse.ArgumentParser(description="StreamingBench reactive streaming QA evaluation")
    parser.add_argument("--run_name", type=str, default="reactive_output_0d5")
    parser.add_argument("--drop_method", type=str, default="feature")
    parser.add_argument("--drop_threshold", type=float, default=0.5)
    parser.add_argument("--drop_relative", action="store_true", help="If set, use relative drop mode (default: absolute)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Local folder with Qwen3-VL weights")
    parser.add_argument("--task_csv", type=str, required=True, help="StreamingBench Real_Time_Visual_Understanding.csv")
    parser.add_argument("--video_dir", type=str, required=True, help="Root folder with sample_*/video.mp4")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory for logs, output/, drop/")
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=1016)
    parser.add_argument("--tolerance_time", type=int, default=2)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument(
        "--sg_generation_interval",
        type=int,
        default=20,
        help="Scene graph every N frames (-1 = every frame; e.g. 3 = every third frame)",
    )
    parser.add_argument(
        "--sg_force_first_frame",
        action="store_true",
        help="If set, always generate a graph on the first frame of the window",
    )
    return parser


def main():
    args = build_parser().parse_args()
    curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.curr_time = curr_time
    args.log_path = osp.join(args.result_dir, "log", f"{args.run_name}_{curr_time}.log")
    args.output_jsonl = osp.join(args.result_dir, "output", f"{args.run_name}_{curr_time}.jsonl")
    args.dr_save_path = osp.join(args.result_dir, "drop", f"{args.run_name}_{curr_time}.jsonl")
    args.drop_absolute = not args.drop_relative

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "output"), exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "drop"), exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "log"), exist_ok=True)

    file_handler = logging.FileHandler(args.log_path)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Running %s on Reactive Streaming Task", args.run_name)
    logger.info("Drop method: %s", args.drop_method)
    logger.info("Drop threshold: %s", args.drop_threshold)
    logger.info("Drop %s", "absolute" if args.drop_absolute else "relative")
    logger.info("Checkpoint path: %s", args.ckpt_path)
    logger.info("Result dir: %s", args.result_dir)
    logger.info("Task csv: %s", args.task_csv)
    logger.info("Video dir: %s", args.video_dir)
    logger.info("Output jsonl: %s", args.output_jsonl)
    logger.info("Drop ratio info save path: %s", args.dr_save_path)
    logger.info("Min pixels: %s", args.min_pixels)
    logger.info("Max pixels: %s", args.max_pixels)
    logger.info("Max frames: %s", args.max_frames)
    logger.info("Min frames: %s", args.min_frames)
    logger.info("Scene Graph generation interval: %s", args.sg_generation_interval)
    logger.info("Scene Graph force first frame: %s", args.sg_force_first_frame)

    eval_r(args)


if __name__ == "__main__":
    main()