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
指导性重缩放的系数：guidance_rescale: float = 0.0, # 指导性重缩放的系数，用于调整模型在某种任务（如生成任务）中使用的指导信号的强度，guidance_rescale 可以调整噪声配置的强度或影响，可能是为了优化生成过程或改善模型输出的质量。[Common Diffusion Noise Schedules and Sample Steps are Flawed]

```

