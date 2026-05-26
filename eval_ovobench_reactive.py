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
from response_graph.Scene_Graph_ovo import Scene_Graph_generation_Real, format_scene_graphs_for_prompt_offline

# Process / CUDA environment (set here or export before launch).
os.environ['NPROC_PER_NODE'] = '1'
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path and hyperparameter defaults are defined in build_parser() / resolved on args in main().

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
        context_window = int(context_window)
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


def build_prompt(task, question, options, _anno_, index, context_Scene_Graphs=None):
    if task in ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        if context_Scene_Graphs is not None:
            context_Scene_Graphs_text = "The relative scene graph shows:\n"
            for timestamp, Scene_Graph in context_Scene_Graphs:
                frame_second_placeholder = f"<{timestamp:.1f} seconds>"
                context_Scene_Graph_text = format_scene_graphs_for_prompt_offline([Scene_Graph])
                context_Scene_Graphs_text += f"{frame_second_placeholder}:{context_Scene_Graph_text}\n"
            formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
            prompt = f"""
                {context_Scene_Graphs_text}

                Question: {question}
                Options:
                {formatted_options}
                Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
                Do not include any additional text or explanation in your response.
            """
        else:
            formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
            prompt = f"""
                Question: {question}
                Options:
                {formatted_options}
                Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
                Do not include any additional text or explanation in your response.
            """

    return prompt


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
    with open(args.task_file, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    # Filter tasks (edit list for your experiment)
    task_list = [task for task in task_list if task.get('task') in ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]]

    # Main evaluation loop.
    with torch.no_grad():
        for item in tqdm(task_list):
            try:
                # Unpack one task row.
                id, video, task, question, options, realtime, gt = \
                    item['id'], item['video'], item['task'], item['question'], item['options'], item['realtime'], item['gt']

                video_path = osp.join(args.video_dir, video)
                # Temporal crop in frame indices.
                start_frame = 0
                end_frame = realtime * args.fps
                # Allocate streaming memory for this clip.
                memories = Memory_Bank_Naive(memory_size = int(end_frame) + 1)
                total_frame = int(end_frame) + 1
                # continue

                args.frame_interval = (total_frame + 499) // 500

                # Streaming Pipeline
                for frame_idx in range(start_frame, int(end_frame) + 1, args.frame_interval):
                    timestamp = frame_idx / args.fps

                    current_frame = extract_frame(video_path, timestamp)
                    if current_frame == None:
                        current_frame = extract_frame(video_path, timestamp - 1)
                        if current_frame == None:
                            current_frame = extract_frame(video_path, timestamp - 2)
                            if current_frame == None:
                                continue
                        # continue
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

                # --- Retrieve graphs for prompt context ---
                current_clip_tokens = memories.get_context_frames(context_window=1)
                current_clip_tokens = torch.cat(current_clip_tokens, dim=1).to(model.device)
                current_Scene_Graph = Scene_Graph_generation_Real(processor, model, current_clip_tokens, question=question)
                if len(current_Scene_Graph.get('scene_graph', [])) != 0:
                    memories.update_graph(timestamp, current_Scene_Graph)

                context_Scene_Graphs = memories.get_newest_graph()
                # context_Scene_Graphs = None

                # --- Build user / assistant prompt embeddings ---
                prompt_response = build_prompt(
                    task=task,
                    question=question,
                    options=options,
                    _anno_=None,
                    index=None,
                    context_Scene_Graphs=context_Scene_Graphs,
                )

                response_ids = processor.tokenizer(
                    ["<|im_start|>user\n" + prompt_response +"<|im_end|>\n"], **text_kwargs)
                response_ids_tensor = torch.tensor(response_ids['input_ids']).to(model.device)
                response_prompt_embeddings = model.get_input_embeddings()(response_ids_tensor)

                past_frame_tokens = memories.get_context_frames(context_window=int((end_frame-start_frame)/args.frame_interval)+1)
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
                if 'current_frame_embeddings' in locals():
                    del current_frame_embeddings
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if response in chr(65 + gt):
                    correct = "yes"
                else:
                    correct = "no"

                output_dict = {
                    'id': id,
                    'video': video,
                    'task': task,
                    'question': question,
                    'response': response,
                    'ground_truth': chr(65 + gt),
                    "correct": correct,
                }

                with open(args.output_jsonl, 'a' if osp.exists(args.output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')

            except Exception as e:
                logger.error(f"Error in processing {item}: {e}")
                print(f"Error in processing {item}: {e}")


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

        return pil_image

    except Exception as e:
        print(f"extract_frame failed: {e}")
        return None


def build_parser():
    parser = argparse.ArgumentParser(description="OVO-Bench reactive evaluation")
    parser.add_argument("--run_name", type=str, default="reactive_output_0d5")
    parser.add_argument("--drop_method", type=str, default="feature")
    parser.add_argument("--drop_threshold", type=float, default=0.5)
    parser.add_argument("--drop_relative", action="store_true", help="If set, use relative drop mode (default: absolute)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Local folder with Qwen3-VL weights")
    parser.add_argument(
        "--task_json",
        "--task_csv",
        dest="task_file",
        type=str,
        required=True,
        help="Path to ovo_bench_new.json (either flag name is accepted)",
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Folder containing benchmark videos (e.g. src_videos)")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory for logs, output/, drop/")
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=720)
    parser.add_argument("--tolerance_time", type=int, default=2)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
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

    logger.info("Running %s on OVO-Bench reactive", args.run_name)
    logger.info("Drop method: %s", args.drop_method)
    logger.info("Drop threshold: %s", args.drop_threshold)
    logger.info("Drop %s", "absolute" if args.drop_absolute else "relative")
    logger.info("Checkpoint path: %s", args.ckpt_path)
    logger.info("Result dir: %s", args.result_dir)
    logger.info("Task json: %s", args.task_file)
    logger.info("Video dir: %s", args.video_dir)
    logger.info("Output jsonl: %s", args.output_jsonl)
    logger.info("Drop ratio info save path: %s", args.dr_save_path)
    logger.info("Min pixels: %s", args.min_pixels)
    logger.info("Max pixels: %s", args.max_pixels)
    logger.info("Max frames: %s", args.max_frames)
    logger.info("Min frames: %s", args.min_frames)

    eval_r(args)


if __name__ == "__main__":
    main()