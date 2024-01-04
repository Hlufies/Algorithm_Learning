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

举例: num_inference_steps = 50, "linspace"
   -> timesteps = [99, 97, 95, 93, 91, 89,
                  87, 85, 83, 81, 79, 77,
                  75, 73, 71, 69, 67, 65,
                  63, 61, 59, 57, 55, 53,
                  51, 48, 46, 44, 42, 40,
                  38, 36, 34, 32, 30, 28,
                  26, 24, 22, 20, 18, 16,
                  14, 12, 10, 8, 6, 4, 2, 0]
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
#### 第5步：准备latents
```
论文里面latents的shape为：[N, 4, 64, 64]
num_channels_latents = self.unet.config.in_channels
latents = self.prepare_latents(
    batch_size * num_images_per_prompt,
    num_channels_latents,
    height,
    width,
    prompt_embeds.dtype,
    device,
    generator,
    latents,
)

def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * self.scheduler.init_noise_sigma
    return latents
```
#### 准备额外的参数
```
extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
```

#### Denoising loop
```
num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
这行代码的作用是计算预热步骤的数量。
这是通过从总的时间步长数量中减去根据推理步骤数量和调度器阶数计算出的值来实现的。
预热步骤在很多算法中都很重要，用于使模型在开始执行主要任务之前达到稳定的状态
```
```
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        --------------------------------------------------------------------------------------------
        # 前面已经说明过， 如果do_classiifier_free_guidance > 1.0, 提示词embedding由两部分拼接起来的
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        # 因此这里的的latents也应是两个相同形状的拼接起来的
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        --------------------------------------------------------------------------------------------

        # 这里将时间步的信息融合到latents里， 这里融合细节另一篇展开剖析
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
        --------------------------------------------------------------------------------------------

        # 通过unet预测noise
        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        --------------------------------------------------------------------------------------------

        # perform guidance
        # 提示词的相关性， 默认值是7.5(CFG)
        if do_classifier_free_guidance:
            # 如果执行提示词引导， 那么noise_pred由两部分组成[uncondition(nevagative), condition]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # 然后最后的noise_pred， 就是无条件预测噪声 + 相关性因子*（noise_pred_text - noise_pred_uncond）
            # noise_pred_text - noise_pred_uncond: 这个差值表示有条件输入和无条件输入生成的噪声之间的差异。这个差异捕捉了条件信息（例如，文本描述）如何改变生成的输出。
            # 通过将无条件噪声与放大的条件差异相加，生成最终的噪声预测。这样，生成的结果既考虑了无条件的基线，又融入了有条件信息的影响。
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        --------------------------------------------------------------------------------------------

        if do_classifier_free_guidance and guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            # 这里执行的重放缩， 暂时没有读论文
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            #---------------------------------------------------------------------------------------

            rescale_noise_cfg 函数的目的是调整扩散模型的噪声配置，以便更好地结合无条件和有条件的信息。
            这样的调整基于对扩散过程的研究，旨在优化生成图像的质量，防止过度曝光，同时保持图像的自然度和丰富性。
            这种方法在基于扩散的生成模型中尤其重要，因为它们依赖于[精确控制噪声过程]来生成高质量的结果

            def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
                # noise_cfg: 噪声配置，可能是模型生成的原始噪声。
                # noise_pred_text: 基于文本（或其他条件信息）的噪声预测。
                # guidance_rescale: 一个缩放因子，用于调整条件信息对最终结果的影响。
                """
                Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
                """

                # 计算 noise_pred_text 在除了第一个维度（通常是批次维度）之外的所有维度上的标准差。
                std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
                # 计算 noise_cfg 在除了第一个维度（通常是批次维度）之外的所有维度上的标准差。
                std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
                # 使用计算出的标准差之比来重新缩放 noise_cfg。这一步是基于特定的扩散模型调整策略，可以帮助纠正过度曝光等问题。
                noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
                # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
                # 将重缩放后的噪声与原始噪声混合。通过 guidance_rescale 控制两者的混合比例，以避免生成的图像看起来过于平淡或无特色。
                noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
                return noise_cfg

            #---------------------------------------------------------------------------------------

        --------------------------------------------------------------------------------------------
        # compute the previous noisy sample x_t -> x_t-1
        # self.scheduler.step 函数被用于在扩散过程中计算前一个噪声样本（从 x_t 到 x_t-1）。
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # call the callback, if provided
        # 召回
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
            progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
```
  
<div style="text-align:center">
    <img width="413" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/089d0e36-8b1c-4d5f-a152-86bde87593db">
</div>

```
def step(
    self,
    # noise_pred
    model_output: torch.FloatTensor,
    # 时间步
    timestep: int,
    # latents
    sample: torch.FloatTensor,
    # 额外的调整参数
    generator=None,
    # 是否返回字典
    return_dict: bool = True,
) -> Union[DDPMSchedulerOutput, Tuple]:
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
    process from the learned model outputs (most often the predicted noise).
    反向SDE预测：该函数通过反向模拟随机微分方程（SDE）来预测前一个时间步的样本。
    在扩散模型中，这通常意味着从添加噪声的样本中逐步去除噪声，从而逐渐恢复出原始的、清晰的数据。
    从模型输出传播：这个过程基于从模型（如深度神经网络）学习到的输出，这些输出通常是对加入样本的噪声的预测。

    Returns:
        [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
            If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
            tuple is returned where the first element is the sample tensor.

    """
    t = timestep

    # 找到上一步的timestep
    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    # Diffsuion公式
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    variance = 0
    if t > 0:
        device = model_output.device
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=device, dtype=model_output.dtype
        )
        if self.variance_type == "fixed_small_log":
            variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
        elif self.variance_type == "learned_range":
            variance = self._get_variance(t, predicted_variance=predicted_variance)
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample,)

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
```


