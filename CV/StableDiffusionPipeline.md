# StableDiffusionPipeline
## 初始化：init 
```
  def __init__(
      self,
      vae: AutoencoderKL,       
      text_encoder: CLIPTextModel,
      tokenizer: CLIPTokenizer,
      unet: UNet2DConditionModel,
      scheduler: KarrasDiffusionSchedulers,
      safety_checker: StableDiffusionSafetyChecker,
      feature_extractor: CLIPImageProcessor,
      requires_safety_checker: bool = True,
  ):
      super().__init__()
  
      ..........
      ..........
    
      # 放缩因子
      self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
      # 图像处理
      self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
      # config
      self.register_to_config(requires_safety_checker=requires_safety_checker)
```
## 主要部分 def __call__
### 函数修饰
```
@torch.no_grad()：
@replace_example_docstring(EXAMPLE_DOC_STRING)：
-----------------------------------------------------------------------------------------------------------------------
@torch.no_grad():是一个专门用于PyTorch的装饰器，用于在函数内部禁用梯度计算。
在训练机器学习模型时需要梯度计算，但在只进行前向传播（如进行预测或评估模型）时不需要梯度。
使用此装饰器会临时将所有的 requires_grad 标志设置为false，这在您不打算调用 .backward() 时可以节省内存和计算资源。
@replace_example_docstring(EXAMPLE_DOC_STRING)：这个装饰器的具体行为取决于 replace_example_docstring 函数的实现细节.
通常，这种类型的装饰器用于动态修改函数的文档字符串。在这个例子中，它可能将函数的文档字符串替换为 EXAMPLE_DOC_STRING 中定义的内容。
这在需要根据不同情况展示不同文档或示例时非常有用。
------------------------------------------------------------------------------------------------------------------------
一下代码(字符串)是huggingface上sd的参考源码
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""
```
### 参数部分
```
提示词：prompt: Union[str, List[str]] = None,
指定图像的高：height: Optional[int] = None,
指定图像的宽：width: Optional[int] = None,
Inference的步数：num_inference_steps: int = 50,
引导词相关性系数：guidance_scale: float = 7.5,
负提示词：negative_prompt: Optional[Union[str, List[str]]] = None,
每个prompt生成图像的数量：num_images_per_prompt: Optional[int] = 1,

eta: float = 0.0,
DDPMScheduler的eta（η）的使用：eta 参数仅在使用 DDIMScheduler（一种特定的调度器）时有效，对于其他类型的调度器，该参数将被忽略。在 DDIM 模型中，eta 用于控制生成过程中的随机性。它的值应该在 [0, 1] 范围内。在这个范围内，较小的 eta 值意味着生成过程更接近确定性，而较大的 eta 值引入更多的随机性。[https://arxiv.org/abs/2010.02502]

生成器：generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
潜空间：latents: Optional[torch.FloatTensor] = None,
提示词的emb：prompt_embeds: Optional[torch.FloatTensor] = None,
负提示词的emb：negative_prompt_embeds: Optional[torch.FloatTensor] = None,
输出的格式：output_type: Optional[str] = "pil",
是否返回字典：return_dict: bool = True,
回调函数：callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
控制调用回调函数的频率：callback_steps: int = 1,
交叉注意力参数：cross_attention_kwargs: Optional[Dict[str, Any]] = None,
指导性重缩放的系数：guidance_rescale: float = 0.0, # 指导性重缩放的系数，用于调整模型在某种任务（如生成任务）中使用的指导信号的强度.
guidance_rescale 可以调整噪声配置的强度或影响，可能是为了优化生成过程或改善模型输出的质量。Guidance rescale factor should fix overexposure when using zero terminal SNR
[Common Diffusion Noise Schedules and Sample Steps are Flawed]
```
### 返回值
```
 [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
如果 return_dict为True，返回stable_diffusion.StableDiffusionPipelineOutput
否则，返回 tuple。元组的第一个元素是一个包含生成图像的列表。元组的第二个元素是一个布尔值列表，每个布尔值指示相应的生成图像是否包含“不适合工作场所”（not-safe-for-work，简称nsfw）内容。
```
### 主要内容
#### 第0步：给Unet默认的高和宽
```
如果调用者没有指定 height 或 width，代码将使用 UNet 配置中的样本大小乘以一个缩放因子来作为默认的尺寸。
self.unet.config.sample_size是config指定，self.vae_scale_factor是放缩因子
height = height or self.unet.config.sample_size * self.vae_scale_factor
width = width or self.unet.config.sample_size * self.vae_scale_factor
```
#### 第1步：检查输入
```
self.check_inputs(prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

图像必须是8的整数倍
if height % 8 != 0 or width % 8 != 0:
    raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
.......
```
#### 第2步：设置召回参数
```
如果prompt不为空且是字符串类型，比如 "hello"
if prompt is not None and isinstance(prompt, str):
    batch_size = 1
如果prompt不为空但是是list类型，比如["hi", "hi"]
elif prompt is not None and isinstance(prompt, list):
    batch_size = len(prompt)
else:
如果prompt为空，那么说明输入之前用其他model embedding了prompt
    batch_size = prompt_embeds.shape[0]
```
```
指定执行的设备
device = self._execution_device
是否执行文本引导
do_classifier_free_guidance = guidance_scale > 1.0
```
#### 第3步：encode prompt
```
text_encoder_lora_scale = (cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None)
prompt_embeds, negative_prompt_embeds = self.encode_prompt(
    prompt,
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    lora_scale=text_encoder_lora_scale,
)
if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
```
```
encode的核心部分
1. tokenizer
text_inputs = self.tokenizer(
      prompt,
      padding="max_length",
      max_length=self.tokenizer.model_max_length,
      truncation=True,
      return_tensors="pt",
  )
2. attention_mask
if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
    attention_mask = text_inputs.attention_mask.to(device)
else:
    attention_mask = None
3. embedding
prompt_embeds = self.text_encoder(
    text_input_ids.to(device),
    attention_mask=attention_mask,
)
prompt_embeds = prompt_embeds[0]
---------------------------------------------------------------------------------------------------------------------------------------------------
在深度学习和自然语言处理（NLP）中，attention_mask 是一个非常重要的概念，尤其是在使用基于注意力机制的模型（如Transformer架构）时。
这段代码中提到的 attention_mask, 区分有效数据和填充数据:
在处理文本数据时，特别是在批处理过程中，为了保持批次中所有样本的大小一致，通常需要对短于最大长度的文本进行填充。这种填充通常用一些特殊的符号（如0或特殊的词汇）来实现。
attention_mask 用于指示哪些部分是真实的数据，哪些部分是为了批处理而添加的填充。在注意力计算中，这可以确保模型只关注真实的数据部分，而忽略填充部分。
在基于Transformer的模型中，attention_mask用于控制注意力机制的聚焦区域。
它告诉模型在进行自注意力（self-attention）计算时应该关注哪些位置（即输入文本中的有效部分），哪些位置应该被忽略（即填充部分）。
通过检查 self.text_encoder.config中的use_attention_mask 配置项来确定是否需要创建和使用attention_mask。
这表示不是所有的模型配置都需要使用 attention_mask，这取决于模型的具体设计和配置。
---------------------------------------------------------------------------------------------------------------------------------------------------

4. 使用num_images_per_prompt这个参数，也就是复制相同的prompt：duplicate text embeddings for each generation per prompt, using mps friendly method
prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
5. uncondition embedding
非条件型：""或者是nagative_prompt
uncond_input = [""]*batch or negative_prompt 
```
#### 第4步：准备时间步数
```
self.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = self.scheduler.timesteps
# "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
---------------------------------------------------------------------------------------------------------------------------------------------------
DDPM主要由三种策略处理timesteps
1. linspace
如果 timestep_spacing 设置为 "linspace"，则使用 np.linspace 在0到 num_train_timesteps - 1之间均匀地生成指定数量 (num_inference_steps) 的时间步长。
这些时间步长被反转（[::-1]），四舍五入并转换为整数。
2. leading
如果设置为 "leading"，则根据 num_train_timesteps 和 num_inference_steps 的比率计算步长。
这里使用 np.arange 生成一系列时间步长，然后乘以步长比率，四舍五入，并加上一个偏移量 steps_offset。
这种方法可能用于在训练过程的早期更密集地采样时间步长
3. trailing
如果设置为 "trailing"，则按照类似的比率逻辑反向生成时间步长。
这里使用 np.arange 从 num_train_timesteps 到 0 生成时间步长，并按比率进行调整。生成的时间步长减去1以匹配索引。
---------------------------------------------------------------------------------------------------------------------------------------------------
```



