# Stable Diffusion Train

<img width="336" alt="image" src="https://github.com/Hlufies/Algorithm_Learning/assets/130231524/47cb1d2b-1809-41a9-a3dd-56cc46915266">

## 1. add noise
```
......
# prepare latents [N, 4, 64, 64], noise(random) [N, 4,64,64], timesteps(random) [N, 1]
noisy_latents = noise_scheduler.add_noise_(latents, noise, timesteps)

def add_noise_(
    self,
    original_samples: torch.FloatTensor,
    noise: torch.FloatTensor,
    timesteps: torch.IntTensor,
) -> torch.FloatTensor:
    # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
    alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
    timesteps = timesteps.to(original_samples.device)

    # 根据公式，计算根号下的alpha_t
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
    while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    # 根据公式，计算根号下的1-alpha_t
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    # 添加噪声的样本为：noise_latents
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples 
......
```
## 2. 将加噪的样本输入进Unet

```
# 得到预测的噪声
noise_pred = unet(noisy_latents.to(accelerator.device), timesteps, encoder_hidden_states_none_train).sample
                                
```
## 3. 计算损失
```
## 使用均方误差， 这里的target也就是noise(random)
loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3])
                            
```
## 4. backward
```
accelerator.backward(loss_sum)
      if accelerator.sync_gradients and args.max_grad_norm != 0.0:
          if train_text_encoder:
              params_to_clip = itertools.chain(unet.parameters(), text_encoder.parameters())
          else:
              params_to_clip = unet.parameters()
          accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad(set_to_none=True)
```
## 说明
```
这里引用的代码是Kohya_ss中的源码
```
