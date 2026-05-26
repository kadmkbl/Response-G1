import torch

text_kwargs = {
    "padding": True,
    "return_token_type_ids": False
}

def encode_images(model, pixel_values, image_grid_thw):
    image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
    token_per_frame = image_embeds[0].shape[0]

    return image_embeds, deepstack_image_embeds, token_per_frame

def whole_current_frame_tokens(processor, model, timestamp, frame_tokens):
    frame_second_placeholder = f"<{timestamp:.1f} seconds>" # Qwen3VL requires frame timestamps to construct prompts
    pre_text = "<|im_start|>user\n" + frame_second_placeholder + processor.vision_start_token
    pre_ids = processor.tokenizer(pre_text, **text_kwargs)
    pre_ids_tensor = torch.tensor(pre_ids['input_ids']).to(model.device)
    pre_embeddings = model.get_input_embeddings()(pre_ids_tensor)

    post_text = processor.vision_end_token + "<|im_end|>\n"
    post_ids = processor.tokenizer(post_text, **text_kwargs)
    post_ids_tensor = torch.tensor(post_ids['input_ids']).to(model.device)
    post_embeddings = model.get_input_embeddings()(post_ids_tensor)

    frame_tokens = torch.cat([pre_embeddings, frame_tokens[0], post_embeddings], dim=0).to(model.device)

    return frame_tokens