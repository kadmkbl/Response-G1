from qwen_vl_utils import process_vision_info
import torch

MIN_PIXELS = 448*448
MAX_PIXELS = 448*448

text_kwargs = {
    "padding": True,
    "return_token_type_ids": False
}

generate_kwargs = {
    "do_sample": False,
    "num_beams": 1, 
    "min_length": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 256,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

# Balance: (1) query relevance for retrieval, (2) faithfulness to visible content, (3) coverage beyond query-only terms.
def Scene_Graph_generation_po_frame(processor, model, current_frame_tokens, question=None, Query_Graph=None):
    """
    Generate Scene Graph for the current video frame
    """
    # terminology_text = None
    if Query_Graph is not None:
        object_list = Query_Graph['object_list']
        object_list_text = ", ".join(object_list)
        relation_list = Query_Graph['relation_list']
        relation_list_text = ", ".join(relation_list)
        terminology_text = f"{object_list_text} {relation_list_text}"
    else:
        object_list = []
        relation_list = []

    prompt = """
        As a scene graph analysis expert, please analyze the current video clip and generate a focused Scene Graph that is most relevant to the user's query.

        **User Query:** "{question}"

        **Analysis Guidelines:**
        - First, objectively describe what is actually present in the clip
        - Never invent objects or relationships that are not actually visible
        - Do NOT include duplicate triples with identical meaning

        **Scene Graph**
        - Format: [Subject, Relation, Object]
        - Include: Objects, their spatial relationships (near, left_of, right_of, above, below, etc), and actions (holding, walking, running, hitting, etc)

        Please only output the scene graph triplets in the following format:
        Scene Graph:
            1. [person, holding, cup]
            2. [dog, running_towards, park]
            3. [car, parked_near, building]
        ...
    """
    formatted_prompt = prompt.format(question=question)

    prompt_ids = processor.tokenizer(
        ["<|im_start|>user\n" + formatted_prompt +"<|im_end|>\n"], **text_kwargs)
    prompt_ids_tensor = torch.tensor(prompt_ids['input_ids']).to(model.device)
    prompt_embeddings = model.get_input_embeddings()(prompt_ids_tensor)

    input_embeddings = torch.cat([current_frame_tokens, prompt_embeddings], dim=1).to(model.device)
    generation_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    generation_ids_tensor = torch.tensor(generation_ids['input_ids']).to(model.device)
    generation_prompt_embeddings = model.get_input_embeddings()(generation_ids_tensor)
    input_embeddings = torch.cat([input_embeddings, generation_prompt_embeddings], dim=1).to(model.device)

    attention_masks = torch.ones(input_embeddings.shape[:2], dtype=torch.long).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            **generate_kwargs,
        )
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    scene_graph = parse_scene_graph_response(response)

    # Release CUDA temporaries
    del prompt_ids_tensor, prompt_embeddings, generation_ids_tensor, generation_prompt_embeddings
    del input_embeddings, attention_masks, generated_ids
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return scene_graph


def Scene_Graph_generation_po_frame_wo_query(processor, model, current_frame_tokens):
    """
    Generate Scene Graph for the current video frame
    """

    prompt = """
        As a scene graph analysis expert, please analyze the current video clip and generate a focused Scene Graph.

        **Analysis Guidelines:**
        - First, objectively describe what is actually present in the clip
        - Never invent objects or relationships that are not actually visible
        - Do NOT include duplicate triples with identical meaning

        **Scene Graph**
        - Format: [Subject, Relation, Object]
        - Include: Objects, their spatial relationships (near, left_of, right_of, above, below, etc), and actions (holding, walking, running, hitting, etc)

        Please only output the scene graph triplets in the following format:
        Scene Graph:
            1. [person, holding, cup]
            2. [dog, running_towards, park]
            3. [car, parked_near, building]
        ...
    """
    formatted_prompt = prompt

    prompt_ids = processor.tokenizer(
        ["<|im_start|>user\n" + formatted_prompt +"<|im_end|>\n"], **text_kwargs)
    prompt_ids_tensor = torch.tensor(prompt_ids['input_ids']).to(model.device)
    prompt_embeddings = model.get_input_embeddings()(prompt_ids_tensor)

    input_embeddings = torch.cat([current_frame_tokens, prompt_embeddings], dim=1).to(model.device)
    generation_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    generation_ids_tensor = torch.tensor(generation_ids['input_ids']).to(model.device)
    generation_prompt_embeddings = model.get_input_embeddings()(generation_ids_tensor)
    input_embeddings = torch.cat([input_embeddings, generation_prompt_embeddings], dim=1).to(model.device)

    attention_masks = torch.ones(input_embeddings.shape[:2], dtype=torch.long).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            **generate_kwargs,
        )
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    scene_graph = parse_scene_graph_response(response)

    # Release CUDA temporaries
    del prompt_ids_tensor, prompt_embeddings, generation_ids_tensor, generation_prompt_embeddings
    del input_embeddings, attention_masks, generated_ids
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return scene_graph


def Query_Graph_generation(processor, model, question):
    """
    Generate Query Graph for the current user query
    """
    prompt = """
        As a scene graph analysis expert, please parse the user query into a structured scene graph that explicitly captures its anticipated visual scene and response conditions.

        **User Query:** "{question}"

        **Analysis Guidelines:**
        - Identify all objects, attributes, and relationships explicitly mentioned or logically implied in the query
        - Focus on elements that define the visual conditions required for a valid response
        - Maintain semantic faithfulness to the original query intent
        - Do NOT include duplicate triples with identical meaning

        **Scene Graph**
        - Format: [Subject, Relation, Object]
        - Include: Objects, their spatial relationships (near, left_of, right_of, above, below, etc), and actions (holding, walking, running, hitting, etc)

        Please only output the scene graph triplets in the following format:
        Scene Graph:
            1. [person, holding, cup]
            2. [dog, running_towards, park]
            3. [car, parked_near, building]
        ...
    """
    formatted_prompt = prompt.format(question=question)

    prompt_ids = processor.tokenizer(
        ["<|im_start|>user\n" + formatted_prompt +"<|im_end|>\n"], **text_kwargs)
    prompt_ids_tensor = torch.tensor(prompt_ids['input_ids']).to(model.device)
    prompt_embeddings = model.get_input_embeddings()(prompt_ids_tensor)

    input_embeddings = torch.cat([prompt_embeddings], dim=1).to(model.device)
    generation_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    generation_ids_tensor = torch.tensor(generation_ids['input_ids']).to(model.device)
    generation_prompt_embeddings = model.get_input_embeddings()(generation_ids_tensor)
    input_embeddings = torch.cat([input_embeddings, generation_prompt_embeddings], dim=1).to(model.device)

    attention_masks = torch.ones(input_embeddings.shape[:2], dtype=torch.long).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_masks,
            **generate_kwargs,
        )
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    response = output_text[0]

    query_graph = parse_scene_graph_response(response)

    object_list_1 = [triplet[0] for triplet in query_graph['scene_graph']]
    object_list_2 = [triplet[2] for triplet in query_graph['scene_graph']]
    object_list = list(set(object_list_1 + object_list_2))
    relation_list = list(set([triplet[1] for triplet in query_graph['scene_graph']]))
    query_graph['object_list'] = object_list
    query_graph['relation_list'] = relation_list

    del prompt_ids_tensor, prompt_embeddings, generation_ids_tensor, generation_prompt_embeddings
    del input_embeddings, attention_masks, generated_ids
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return query_graph


def parse_scene_graph_response(response):
    """
    Parse the model-generated response to extract Scene Graph
    """
    scene_graph = {
        'scene_graph': [],
        # 'frame_description': ""
    }
    
    lines = response.split('\n')
    current_section = None
    in_scene_graph_section = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        if line.startswith('Scene Graph'):
            current_section = 'scene_graph'
            in_scene_graph_section = True
            continue
        elif line == '...':
            in_scene_graph_section = False
            continue
        
        # Parse Scene Graph
        if current_section == 'scene_graph' and in_scene_graph_section and line:
            if line.startswith(tuple(str(i) + '.' for i in range(1, 100))):
                # Remove numbering
                line = line.split('.', 1)[1].strip()
            
            if line.startswith('[') and line.endswith(']'):
                try:
                    # Parse triplet [subject, relation, object]
                    triplet_str = line[1:-1]  # Remove brackets
                    parts = [part.strip() for part in triplet_str.split(',')]
                    if len(parts) >= 3:
                        # Clean and validate the triplet
                        clean_triplet = []
                        for part in parts[:3]:
                            clean_part = ' '.join(part.split()).strip()
                            if clean_part:  # Only add non-empty parts
                                clean_triplet.append(clean_part)
                        
                        # Ensure we have all three components
                        if len(clean_triplet) == 3:
                            scene_graph['scene_graph'].append(clean_triplet)
                        else:
                            print(f"Incomplete triplet: {clean_triplet}")
                except Exception as e:
                    print(f"Failed to parse scene graph triplet: {line}, error: {e}")
    
    return scene_graph


def format_scene_graphs_for_prompt(scene_graphs):
    """Format scene-graph dicts as compact text for prompts."""
    formatted = []
    # recent_graphs = scene_graphs[-3:] if len(scene_graphs) > 3 else scene_graphs  # optional: cap prompt length
    recent_graphs = scene_graphs[:]

    for i, scene_graph in enumerate(recent_graphs):
        # formatted.append(f"Scene Graph:")
        triplets = scene_graph.get('scene_graph', [])
        # for j, triplet in enumerate(triplets[:5]):  # optional: cap triplets per graph
        for j, triplet in enumerate(triplets[:]):
            subject, relation, obj = triplet
            formatted.append(f" - {subject} {relation} {obj}")
        # if len(triplets) > 5:
        #     formatted.append(f"  ... and {len(triplets) - 5} more relations")

    return "\n".join(formatted)


def format_scene_graphs_for_prompt_query(scene_graphs):
    """Format scene-graph dicts as compact text for prompts."""
    formatted = []
    # recent_graphs = scene_graphs[-3:] if len(scene_graphs) > 3 else scene_graphs  # optional: cap prompt length
    recent_graphs = scene_graphs[:]

    for i, scene_graph in enumerate(recent_graphs):
        # formatted.append(f"Scene Graph:")
        triplets = scene_graph.get('scene_graph', [])
        # for j, triplet in enumerate(triplets[:5]):  # optional: cap triplets per graph
        for j, triplet in enumerate(triplets[:]):
            subject, relation, obj = triplet
            formatted.append(f" - {subject} {relation} {obj}")
        # if len(triplets) > 5:
        #     formatted.append(f"  ... and {len(triplets) - 5} more relations")

    return "\n".join(formatted)


def format_scene_graphs_for_prompt_offline(scene_graphs):
    """Format scene-graph dicts as compact text for prompts."""
    formatted = []
    seen_triplets = set()  # Dedup identical triplets
    # recent_graphs = scene_graphs[-3:] if len(scene_graphs) > 3 else scene_graphs  # optional: cap prompt length
    recent_graphs = scene_graphs[:]

    for i, scene_graph in enumerate(recent_graphs):
        # formatted.append(f"Scene Graph:")
        triplets = scene_graph.get('scene_graph', [])
        # for j, triplet in enumerate(triplets[:5]):  # optional: cap triplets per graph
        for j, triplet in enumerate(triplets[:]):
            subject, relation, obj = triplet
            # Tuple key for set deduplication
            triplet_tuple = (subject, relation, obj)
            if triplet_tuple not in seen_triplets:
                seen_triplets.add(triplet_tuple)
                formatted.append(f" - {subject} {relation} {obj}")
        # if len(triplets) > 5:
        #     formatted.append(f"  ... and {len(triplets) - 5} more relations")

    return "\n".join(formatted)


def format_scene_graphs_for_prompt_query_offline(scene_graphs):
    """Format scene-graph dicts as compact text for prompts."""
    formatted = []
    seen_triplets = set()  # Dedup identical triplets
    # recent_graphs = scene_graphs[-3:] if len(scene_graphs) > 3 else scene_graphs  # optional: cap prompt length
    recent_graphs = scene_graphs[:]

    for i, scene_graph in enumerate(recent_graphs):
        # formatted.append(f"Scene Graph:")
        triplets = scene_graph.get('scene_graph', [])
        # for j, triplet in enumerate(triplets[:5]):  # optional: cap triplets per graph
        for j, triplet in enumerate(triplets[:]):
            subject, relation, obj = triplet
            # Tuple key for set deduplication
            triplet_tuple = (subject, relation, obj)
            if triplet_tuple not in seen_triplets:
                seen_triplets.add(triplet_tuple)
                formatted.append(f" - {subject} {relation} {obj}")
        # if len(triplets) > 5:
        #     formatted.append(f"  ... and {len(triplets) - 5} more relations")

    return "\n".join(formatted)