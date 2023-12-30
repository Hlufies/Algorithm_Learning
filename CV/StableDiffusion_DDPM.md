# DDPM

code版

# 1. Timesteps

```python
def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """

    print('Timesteps===================================================')
    '''
    print(f'timestpes : {timesteps}')
    print(f'timesteps shape : {timesteps.shape}')
    print(f'embeding_dim : {embedding_dim}')
    print(f'flip_sin_to_cos : {flip_sin_to_cos}')
    print(f'downscale_freq_shift : {downscale_freq_shift}')
    print(f'scale : {scale}')
    print(f'max_period : {max_period}')
    timestpes : tensor([ 90, 867], device='cuda:0')
    timesteps shape : torch.Size([2])
    embeding_dim : 320
    flip_sin_to_cos : True
    downscale_freq_shift : 0   
    scale : 1
    max_period : 10000
    
    downscale_freq_shift用于调整频率的分布范围。在您的例子中，此值为0，意味着没有对频率分布进行额外的下调或平移。如果这个值不是0，它将影响指数的分布，进而影响正弦和余弦嵌入的频率特性。
    scale的值为1，这意味着生成的嵌入将保持原始计算值，不会被缩放。如果scale的值不同于1，它将按照给定的比例因子放大或缩小嵌入。
    max_period的对数值决定了频率范围的下限。较大的max_period值会导致更低的频率范围，从而使嵌入能够在较低频率上变化。这对于模型捕捉和表示时间步的不同阶段是重要的。
    '''

    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    # 将嵌入维度除以2，得到半维度，用于创建正弦和余弦嵌入
    half_dim = embedding_dim // 2 

    #计算过程:
    # exponent是通过首先生成一个等差数列（从0开始，以1为步长，直到达到嵌入维度的一半），然后将每个数值乘以-math.log(max_period)来计算的。
    # 接着，这些值被除以(half_dim - downscale_freq_shift)。在您的代码中，由于downscale_freq_shift为0，所以实际上是除以half_dim。
    # 意义:
    # exponent代表了一系列递减的值，这些值用于生成正弦和余弦波形的频率。在您的代码中，这些波形用于表示不同的时间步。
    # 通过使用指数函数，可以为每个时间步生成一个唯一的多维嵌入。这些嵌入通过正弦和余弦函数的不同频率组合而成，从而捕获时间步的特征。
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    #
    # 在这个图表中，我们展示了两个示例时间步（90和867）的嵌入值。每个点代表嵌入向量中的一个维度的值。

    # 横轴表示嵌入向量中的维度。
    # 纵轴表示每个维度的嵌入值。

    # 可以看到，对于两个不同的时间步，嵌入值有显著的差异。
    # 这表明每个时间步都被成功地转换成了一个独特的多维嵌入，
    # 其模式随时间步的变化而变化。
    # 这种差异化的表示对于模型能够识别和利用时间信息至关重要，
    # 尤其是在涉及复杂时间动态的生成模型中。
    emb = torch.exp(exponent)

    # 这一行将时间步（timesteps）与之前计算的指数值（emb）相乘。
    # timesteps[:, None]将时间步张量转换为列向量（通过增加一个新的维度），使其可以与emb（行向量）进行广播乘法。
    # 结果是，每个时间步都与一系列指数值相乘，生成了一个独特的嵌入向量。
    # print(timesteps[:, None].shape) # [2,1]
    # print(emb[None, :].shape) # [1, 160]
    emb = timesteps[:, None].float() * emb[None, :] # [2, 160]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings    
    # 这行代码的作用是将通过正弦函数 (torch.sin) 和余弦函数 (torch.cos) 处理后的嵌入值进行拼接，从而创建最终的时间步嵌入。
    # 具体来说：
    # torch.sin(emb) 和 torch.cos(emb): 这两个操作分别计算之前步骤中得到的每个嵌入值的正弦和余弦值。由于emb包含了经过指数和时间步调整的一系列值，这些正弦和余弦操作将这些值转换成周期性的波形。
    # torch.cat([...], dim=-1): 此操作将正弦和余弦计算的结果沿最后一个维度（dim=-1）拼接起来。
		# 如果emb是一个二维张量（例如，批量大小为N和嵌入维度为M），则结果将是一个形状为 [N, 2M] 的张量，其中前一半是正弦值，后一半是余弦值。
    # 这样，每个时间步的嵌入就包含了对应时间步数值的正弦和余弦表示，这种表示形式有助于模型捕捉和利用时间信息。通过结合正弦和余弦波形，模型可以更精确地表示和理解不同时间步之间的变化。
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # [2, 320]

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    
    # zero pad
    
    # 这行代码是为了处理嵌入维度为奇数的情况。
    # 当嵌入维度（embedding_dim）是奇数时，由于正弦和余弦嵌入的组合会导致嵌入向量的维度翻倍，这可能导致最终的维度是奇数。
    # 为了保持嵌入向量的维度为偶数，代码在需要时对嵌入向量进行了零填充（padding）。
    # 具体来说：
    # if embedding_dim % 2 == 1:：检查嵌入维度是否为奇数。如果是，执行下一行代码。
    # emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))：对嵌入向量进行零填充。
    # pad函数的参数(0, 1, 0, 0)表示在嵌入向量的最后一个维度的末尾添加一个零。
    # 这样，如果原始的emb维度是奇数，添加一个零后，其维度将变成偶数。
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    print('Timesteps===================================================')
    return emb
```

### 1.1 Exponent value: linear relationship

![Untitled](DDPM%207c7422248a07499f86c9cf75888762c8/Untitled.png)

### 1.2 emb value

在这个图表中，我们展示了两个示例时间步（90和867）的嵌入值。每个点代表嵌入向量中的一个维度的值。

- 横轴表示嵌入向量中的维度。
- 纵轴表示每个维度的嵌入值。

您可以看到，对于两个不同的时间步，嵌入值有显著的差异。这表明每个时间步都被成功地转换成了一个独特的多维嵌入，其模式随时间步的变化而变化。这种差异化的表示对于模型能够识别和利用时间信息至关重要，尤其是在涉及复杂时间动态的生成模型中。

![Untitled](DDPM%207c7422248a07499f86c9cf75888762c8/Untitled%201.png)

![Untitled](DDPM%207c7422248a07499f86c9cf75888762c8/Untitled%202.png)

### 1.3 COS_SIN

![Untitled](DDPM%207c7422248a07499f86c9cf75888762c8/Untitled%203.png)

![Untitled](DDPM%207c7422248a07499f86c9cf75888762c8/Untitled%204.png)

# 2. CFG

```python
noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
              
# perform guidance
if do_classifier_free_guidance:
    # noise_pred.chunk(2) 将 noise_pred 分割成两部分：
    # noise_pred_uncond（无条件的噪声预测）
    # noise_pred_text（文本条件的噪声预测）
    # 这是基于假设 noise_pred 包含两个相同大小的部分，分别对应无条件生成和条件生成。
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

--------------------------------------------------------------------------------------------

**无条件和条件生成的结合：

noise_pred_uncond 表示无条件生成的部分，即不依赖于特定文本或条件的噪声预测。
noise_pred_text 表示条件生成的部分，即依赖于特定文本或条件的噪声预测。
引导尺度（Guidance Scale）：

guidance_scale 是一个超参数，用于调整条件生成相对于无条件生成的影响力。高 guidance_scale 值增强了条件生成部分的影响，使生成的内容更紧密地遵循给定的条件（如特定的文本描述）。
平衡条件和无条件生成：

noise_pred_text - noise_pred_uncond 计算条件生成和无条件生成之间的差异。这一差异反映了条件信息对噪声预测的影响。
将这一差异乘以 guidance_scale，然后加回到无条件噪声预测上，实际上是在加强条件部分的特征，同时保留一定量的无条件生成特征。
改善生成质量：

这种方法允许模型在保持一定随机性和自然性的同时，更好地遵循给定的条件。对于图像生成任务，这意味着生成的图像既符合文本描述，又保持了一定的多样性和自然度。
为什么有效
灵活控制：通过调整 guidance_scale，可以灵活控制条件影响的程度，从而在遵循文本提示的精确性和图像的自然多样性之间找到平衡。
适应不同场景：这种方法在需要根据文本提示生成具体内容，同时希望保持生成结果自然和多样的场景中特别有效。**
```
