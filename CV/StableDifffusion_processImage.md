## VaeImageProcessor

## processImage
```
def preprocess(self,image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],height: Optional[int] = None,width: Optional[int] = None,) -> torch.Tensor:
    # 支持处理的格式PIL, numpy, tensor
    supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)
    .......
```

## post-processImage

```
def postprocess(self,image: torch.FloatTensor,output_type: str = "pil",do_denormalize: Optional[List[bool]] = None,):
    .......
    # 返回格式是latent
    if output_type == "latent":
        return image
    # 返回格式是tensor
    if output_type == "pt":
        return image
    image = self.pt_to_numpy(image)
    # 返回格式是numpy
    if output_type == "np":
        return image
    # 返回格式pil
    if output_type == "pil":
        return self.numpy_to_pil(image)
```


## numpy -> PIL.Image.Image
```
def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    """
    将numpy数组或一批numpy数组转换为PIL图像对象的列表。
    """
    if images.ndim == 3:
        images = images[None, ...]  # 在开头增加一个批次维度
    images = (images * 255).round().astype("uint8")  # 缩放并转换为uint8类型
    pil_images = []
    if images.shape[-1] == 1:
        # 单通道灰度图的特殊处理
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        # 默认为RGB或RGBA图像
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images  # 返回PIL图像对象的列表

```
## PIL.Image.Image -> np.ndarray
```
def pil_to_numpy(images: Union[List[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
    """
    将PIL图像或PIL图像列表转换为NumPy数组。
    """
    # 如果images不是列表，则将其转换为列表
    if not isinstance(images, list):
        images = [images]
    # 将PIL图像转换为NumPy数组，并进行标准化处理
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    # 将图像列表堆叠成一个多维数组
    images = np.stack(images, axis=0)

    return images
```
## numpy -> pytorch tensor
```
@staticmethod
# 定义为一个静态方法，通常是因为您想将这个函数放到一个类中，而不需要这个函数访问类的任何实例属性或方法
def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """
    Convert a NumPy image to a PyTorch tensor.
    """
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images
```
## pytorch tensor -> numpy
```
@staticmethod
def pt_to_numpy(images: torch.FloatTensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images
```
## 图像归一化
```
def normalize(images):
    """
    input: images的数值范围已经是[0,1], 也就是原始image/255
    Normalize an image array to [-1,1].
    """
    return 2.0 * images - 1.0
```
## 图像反归一化
```
@staticmethod
def denormalize(images):
    """
    input: images的数值范围已经是[-1,1]
    Denormalize an image array to [0,1].
    """
    return (images / 2 + 0.5).clamp(0, 1)
```

## 图像转为rgb通道
```
@staticmethod
def convert_to_rgb(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to RGB format.
    """
    image = image.convert("RGB")
    return image
```
## 图像转为灰域
```
def convert_to_grayscale(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts a PIL image to grayscale format.
    """
    image = image.convert("L")
    return image
```
## 获得图像的宽高
```
get_default_height_width
```
## 图像resize
```
def resize(
    self,
    image: [PIL.Image.Image, np.ndarray, torch.Tensor],
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> [PIL.Image.Image, np.ndarray, torch.Tensor]:
```
## 处理成二值图像，一般是用于做mask的
```
def binarize(self, image: PIL.Image.Image) -> PIL.Image.Image:
    """
    create a mask
    """
    image[image < 0.5] = 0
    image[image >= 0.5] = 1
    return image
```
