为什么计算Loss之后的变量值不能在应用了？
```
with torch.no_grad():
                
                latents_pred = latents_pred.to(device).to(weight_dtype)               
                # pred_image = vae.decode(latents_pred).sample.to(device).to(weight_dtype)
                pred_image = decode_image(latents_pred,vae)
                print(pred_image)
                copy_pred_image = copy.deepcopy(pred_image)
                
                # pred_image_numpy = pred_image.cpu().numpy()
                # pred_image = torch_to_numpy(pred_image)
                vae_image = decode_image(ori_latents, vae)
                # vae_image = torch_to_numpy(vae_image)
                # loss3 = F.binary_cross_entropy(pred_image, batch['images'].to(weight_dtype))  + 0.1 * F.binary_cross_entropy(vae_image, batch['images'].to(weight_dtype)) 
                # vae_image = Inverse_Transform(vae_image)
                # pred_image = Inverse_Transform(pred_image)
                # vae_image = INVERSE_TRANSFORM(vae_image)
                # pred_image = INVERSE_TRANSFORM(pred_image)
```
