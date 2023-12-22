```
def get_weighted_text_embeddings(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    device,
    max_embeddings_multiples: Optional[int] = 3,
    no_boseos_middle: Optional[bool] = False,
    clip_skip=None,
):
    """
    根据给定的文本提示，生成加权文本嵌入。

    参数:
        tokenizer: 用于将文本转换为令牌的分词器。
        text_encoder: 文本编码器，用于将令牌转换为嵌入。
        prompt: 用于指导图像生成的文本提示，可以是单个字符串或字符串列表。
        device: 计算设备，如 CPU 或 GPU。
        max_embeddings_multiples: 提示嵌入的最大长度，相对于文本编码器最大输出长度的倍数，默认为3。
        no_boseos_middle: 如果文本令牌长度是文本编码器容量的倍数，是否保留中间每个块的起始和结束令牌，默认为 False。
        clip_skip: 跳过的剪辑长度，用于调整编码。

    返回:
        加权且规范化的文本嵌入。
    """

    # 计算最大嵌入长度
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # 将单个字符串提示转换为列表
    if isinstance(prompt, str):
        prompt = [prompt]

    # 获取带权重的提示和相应的权重
    prompt_tokens, prompt_weights = get_prompts_with_weights(tokenizer, prompt, max_length - 2)

    # 计算最大嵌入长度
    max_length = max([len(token) for token in prompt_tokens])
    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # 填充令牌和权重
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=tokenizer.model_max_length,
    )

    # 将令牌转换为张量
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)

    # 获取未加权的文本嵌入
    text_embeddings = get_unweighted_text_embeddings(
        tokenizer,
        text_encoder,
        prompt_tokens,
        tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )

    # 将权重转换为张量
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=device)

    # 为提示赋予权重并规范化嵌入
    previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * prompt_weights.unsqueeze(-1)
    current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
    text_embeddings = text_embeddings * (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    # 返回加权且规范化的文本嵌入
    return text_embeddings


```
