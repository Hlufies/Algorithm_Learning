# StableDiffusionPipeline
## init
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
