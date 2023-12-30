# UNet2DConditionModel
## 初始化
```
    A conditional 2D UNet model that takes a noisy sample, conditional state, and a timestep and returns a sample shaped output.This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented for all models (such as downloading or saving).
    1. 这里提到的是一个 UNet 模型，它是一种常用于图像处理任务的卷积神经网络架构，特别是在图像分割领域。
    2. "2D" 指的是这个模型处理的是二维数据，如普通的图像。
    3. "条件式"（conditional）可能意味着这个模型的输出不仅依赖于输入图像，还依赖于某些额外的条件或参数。
    4. 输出是与输入样本形状相同的
    5. 这个 UNet 模型是从一个名为 ModelMixin 的类继承而来
    6. 模型的输入。它接收三种输入：
        一个带噪声的样本（可能是图像或其它类型的二维数据），
        一个条件状态（这可能是一个额外的信息，用于指导或改变模型的行为），
        一个时间步（通常用于序列数据或动态过程，可能在这里用于指定处理的特定阶段）
```

### Parameters:
 
1. 样本是输入大小：height和width。sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`): Height and width of input/output sample.  
2. 样本的输入通道：in_channels (`int`, *optional*, defaults to 4): Number of channels in the input sample.  
3. 样本的输出通道：out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.  
4. 是否进行中心化处理：就让让数据范围在[-1,1]之间。center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.  
5. 是否进行将time_embeding进行正弦和余弦翻转。因为time_embedding是由从一个正弦余弦组合的映射而来的，默认的组合是sin-cos。flip_sin_to_cos (`bool`, *optional*, defaults to `False`): Whether to flip the sin to cos in the time embedding.  
6. 是否对time_embedding进行频移。freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.  
7. Unet中降采样模块的类型。默认情况下，元组包含四个元素，前三个是 "CrossAttnDownBlock2D"，最后一个是 "DownBlock2D"。down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`): The tuple of downsample blocks to use.
8. Unet中的中间架构类型。 mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            Block type for middle of UNet, it can be either `UNetMidBlock2DCrossAttn` or
            `UNetMidBlock2DSimpleCrossAttn`. If `None`, the mid block layer is skipped.  
9. Unet中的上采样模块类型。up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`): The tuple of upsample blocks to use.  
````          
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        num_attention_heads (`int`, *optional*):
            The number of attention heads. If not defined, defaults to `attention_head_dim`
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for ResNet blocks (see [`~models.resnet.ResnetBlock2D`]). Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        addition_time_embed_dim: (`int`, *optional*, defaults to `None`):
            Dimension for the timestep embeddings.
        num_class_embeds (`int`, *optional*, defaults to `None`):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, defaults to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        time_embedding_dim (`int`, *optional*, defaults to `None`):
            An optional override for the dimension of the projected time embedding.
        time_embedding_act_fn (`str`, *optional*, defaults to `None`):
            Optional activation function to use only once on the time embeddings before they are passed to the rest of
            the UNet. Choose from `silu`, `mish`, `gelu`, and `swish`.
        timestep_post_act (`str`, *optional*, defaults to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, defaults to `None`):
            The dimension of `cond_proj` layer in the timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            `class_embed_type="projection"`. Required when `class_embed_type="projection"`.
        class_embeddings_concat (`bool`, *optional*, defaults to `False`): Whether to concatenate the time
            embeddings with the class embeddings.
        mid_block_only_cross_attention (`bool`, *optional*, defaults to `None`):
            Whether to use cross attention with the mid block when using the `UNetMidBlock2DSimpleCrossAttn`. If
            `only_cross_attention` is given as a single boolean and `mid_block_only_cross_attention` is `None`, the
            `only_cross_attention` value is used as the value for `mid_block_only_cross_attention`. Default to `False`
            otherwise.
```

# UNet2DConditionOutput
最后返回的数据格式：(batch_size, num_channels, height, width)， FloatTensor
```
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None

```
