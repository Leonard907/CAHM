import re
from concurrent.futures import ThreadPoolExecutor

def remove_citations(text):
    """Removes citations in various formats from a text:
    - [n] where n is a number
    - [n,m,k] where n,m,k are comma-separated numbers
    - [n-m] where n,m are numbers indicating a range

    Args:
        text: The input text containing citations.

    Returns:
        The text with citations removed.
    """
    # Regular expressions to match different citation formats
    patterns = [
        r"\s*\[\d+\]",  # [n]
        r"\s*\[\d+(?:,\d+)*\]",  # [n,m,k]
        r"\s*\[\d+-\d+\]"  # [n-m]
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text)
    return cleaned_text

def summary_postprocess(summary):
    if 'summary' in summary.split('\n\n')[0] and len(summary.split('\n\n')) > 1:
        # remove the first, merge the rest
        summary = '\n\n'.join(summary.split('\n\n')[1:])
    # replace \n\n with space
    return summary.replace('\n\n', ' ')

def post_process_gemma(summary):
    summary = summary_postprocess(summary)
    lines = summary.split('\n')
    # Remove empty lines and strip
    lines = [line.strip() for line in lines if line.strip()]
    return " ".join(lines)

def split_by_labels(text):
    """
    Extract citation labels from text in various formats.
    
    Args:
    text (str): Input text containing citation labels
    
    Returns:
    tuple: A tuple containing:
        - The text with citation labels removed
        - A set of all extracted citation labels
    """
    # Remove spaces inside brackets first
    text = re.sub(r'\[(\s*\d+(?:\s*,\s*\d+)*\s*)\]', lambda m: f'[{m.group(1).replace(" ", "")}]', text)
    # Collect all labels
    labels = []
    
    # Find and process all citation labels
    def process_label_group(label_group):
        # Handle single label format [n]
        if label_group.isdigit():
            return [label_group]
        
        # Handle comma-separated format [m,n,k]
        if ',' in label_group:
            comma_labels = label_group.split(',')
            return comma_labels
        
        # Handle range format [m-n]
        if '-' in label_group:
            start, end = map(int, label_group.split('-'))
            return [str(i) for i in range(start, end + 1)]
        
        return []
    
    # Replace citations and collect labels
    def replace_citation(match):
        label_group = match.group(1)
        group_labels = process_label_group(label_group)
        labels.extend(group_labels)
        return ''
    
    # Remove citations and collect labels
    clean_text = re.sub(r'\[(\d+(?:[-,]\d+)*)\]', replace_citation, text)
    
    return clean_text, labels

def add_label(label, inc):
    # Remove brackets and split by commas
    numbers = [int(num) for num in label.strip('[]').split(',')]
    
    # Add the value to each number
    new_numbers = [num + inc for num in numbers]
    
    # Join the numbers back into a string and add brackets
    return '[' + ','.join(map(str, new_numbers)) + ']'

def parse_non_ascii(text):
    # Regular expression pattern to match non-ASCII characters
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]+')
    
    # Replace all matches with a single space
    cleaned_text = non_ascii_pattern.sub(' ', text)
    
    return cleaned_text

def chunk_texts(document, tokenizer, chunk_size):
    ids = tokenizer(document)['input_ids']
    chunks = []
    for i in range(0, len(ids), chunk_size):
        chunks.append(ids[i:i+chunk_size])
    if len(chunks) > 1 and len(ids) % chunk_size < chunk_size // 2:
        # Combine last two chunks and split evenly
        combined = chunks[-2] + chunks[-1]
        half_point = len(combined) // 2
        chunks[-2] = combined[:half_point]
        chunks[-1] = combined[half_point:]
    chunks = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
    return chunks

def get_top_attr(attr_texts, response, topk):
    _, labels = split_by_labels(response)
    label_counts = {}
    for label in labels:
        indices = [int(idx) for idx in label.strip('[]').split(',')]
        for idx in indices:
            label_counts[idx] = label_counts.get(idx, 0) + 1
    # Remove labels that are not in the attr_texts
    label_counts = {idx - 1: count for idx, count in label_counts.items() if idx <= len(attr_texts)}
    
    # Convert to list of tuples and sort by count in descending order
    label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    if len(label_counts) <= topk:
        return [attr_texts[i] for i in [l[0] for l in label_counts]]
    selected_lc = label_counts[:topk]

    # Check if the last count is present in the rest
    same_count = None
    try:
        if selected_lc[-1][1] == label_counts[topk][1]:
            same_count = selected_lc[-1][1]
            while len(selected_lc) > 0 and selected_lc[-1][1] == label_counts[topk][1]:
                del selected_lc[-1]
    except Exception as e:
        import pdb; pdb.set_trace()
    
    # Fill the rest based on coverage
    coverage_sections = [0.1, 0.3, 0.5, 0.7, 0.9]
    for label, count in selected_lc:
        pos = label / len(attr_texts)
        # find the coverage section closest to pos, and delete the closest section
        closest_section = min(coverage_sections, key=lambda x: abs(x - pos))
        coverage_sections.remove(closest_section)
        label_counts.remove((label, count))

    if same_count is not None:
        same_count_lc = [lc for lc in label_counts if lc[1] == same_count]
        while len(coverage_sections) > 0:
            lc_min_dist = []
            for lc in same_count_lc:
                pos = lc[0] / len(attr_texts)
                min_section = min(coverage_sections, key=lambda x: abs(x - pos))
                lc_min_dist.append((lc, min_section, abs(min_section - pos)))
            closest_section = min(lc_min_dist, key=lambda x: x[2])
            coverage_sections.remove(closest_section[1])
            selected_lc.append(closest_section[0])
            label_counts.remove(closest_section[0])
            same_count_lc.remove(closest_section[0])

    selected_labels = [l[0] for l in selected_lc]
    return [attr_texts[i] for i in selected_labels]

def chunk_texts_with_attribution(document, tokenizer, chunk_size, attribution_chunk_size):
    ids = tokenizer(document)['input_ids']
    chunks = []
    for i in range(0, len(ids), chunk_size):
        chunks.append(ids[i:i+chunk_size])
    if len(chunks) > 1 and len(ids) % chunk_size < chunk_size // 2:
        combined = chunks[-2] + chunks[-1]
        half_point = len(combined) // 2
        chunks[-2] = combined[:half_point]
        chunks[-1] = combined[half_point:]
    # Add attribution to each chunk
    attr_chunks = []
    attr_texts = []
    for c in chunks:
        attr_chunk, attr_text = cite_text(c, attribution_chunk_size, tokenizer)
        attr_chunks.append(attr_chunk)
        attr_texts.append(attr_text)
    return attr_chunks, attr_texts

def cite_text(tokens, cite_size, tokenizer):
    attr_num = max(1, len(tokens) // cite_size)
    attr_size = len(tokens) // attr_num
    attr_chunk = []
    attr_texts = []
    label_count = 1
    for i in range(attr_num-1):
        attr_chunk.append(tokenizer.decode(tokens[i*attr_size:(i+1)*attr_size], skip_special_tokens=True) + f' [{label_count}]')
        attr_texts.append(tokenizer.decode(tokens[i*attr_size:(i+1)*attr_size], skip_special_tokens=True))
        label_count += 1
    attr_chunk.append(tokenizer.decode(tokens[(attr_num-1)*attr_size:], skip_special_tokens=True) + f' [{label_count}]')
    attr_texts.append(tokenizer.decode(tokens[(attr_num-1)*attr_size:], skip_special_tokens=True))
    return "\n".join(attr_chunk), attr_texts

def truncate_text(tokenizer, text, max_length):
    # Tokenize the text
    tokens = tokenizer(text)
    
    # Truncate the tokens to the maximum length
    truncated_tokens = tokens['input_ids'][:max_length]
    
    # Convert the tokens back to text
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
    
    return truncated_text

def parse_qa_file(lines):
    qa_pairs = []
    for l in lines:
        if l.startswith('Q: '):
            question = l[3:]
        elif l.startswith('A: '):
            answer = l[3:]
            qa_pairs.append((question, answer))
    return qa_pairs

def vllm_generate(llm, sampling_params, input_text, mode="single"):
    if mode == "single":
        messages = [
            {"role": "user", "content": input_text}
        ]
        chat_messages = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = llm.generate([chat_messages], sampling_params, use_tqdm=False)
        return summary_postprocess(response[0].outputs[0].text)
    else:
        all_messages = []
        for text in input_text:
            messages = [
                {"role": "user", "content": text}
            ]
            chat_messages = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_messages.append(chat_messages)
        responses = llm.generate(all_messages, sampling_params, use_tqdm=False)
        return [summary_postprocess(r.outputs[0].text) for r in responses]

def llama_generate(model, tokenizer, input_text, max_new_tokens):
    messages = [
        {"role": "user", "content": input_text}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        eos_token_id=terminators,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0,
    )
    output_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return summary_postprocess(output_text)
