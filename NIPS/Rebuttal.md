Dear Reviewer z2Ei, thank you very much for your careful review of our paper and your thoughtful comments. We hope that the following responses will help clarify any potential misunderstandings and alleviate your concerns.

**Q1:** Regarding “Moreover, we implement layer-wise guidance dropout by selectively zeroing out portions of \( s_{1:m} \), thereby diminishing the decoder’s dependency on sub-vector correlations.” Is there existing literature that supports this conclusion? Could you provide a more detailed explanation?

**R1:** Thank you for your comments and we do understand your concerns. To further alleviate your concerns, we provide more explanations.
- Firstly, **our goal is to achieve a bidirectional mapping between images and disentangled variables \( s_{1:m} \).** By reducing the co-adaptations between Unet layers, it enhances the model's generalization ability, meaning that neurons are less likely to rely too much on each other, thereby achieving decoupling. The dropout guided by zeroing out disentangled variables during training essentially aims to encourage the model to obtain linearly independent solutions for \( s_{1:m} \).

- Second, the latest SOTA paper SODA[1] (presented at CVPR 2024, a self-supervised diffusion model designed for representation learning) suggests that disentangled latent spaces can better represent the generated images. **In the conclusion of the reference[1]: To improve localization and reduce correlations among the \( m \) sub-vectors, we present layer masking – a layer-wise generalization of classifier-free guidance [2].** The reference provides ample ablation experiments to validate it.

- Third, we have also thoroughly validated the correctness of this conclusion in our experiments. The results presented in Table 1, Table 2, and Figure 3 of our main experiments in the paper demonstrate the rationale behind this conclusion.

[1] Hudson D A, Zoran D, Malinowski M, et al. Soda: Bottleneck diffusion models for representation learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 23115-23127.

[2] Ho J, Salimans T. Classifier-free diffusion guidance[J]. arXiv preprint arXiv:2207.12598, 2022.

**Q2:** How does the proposed method handle the increasing number of protected entities? Does it require retraining with each addition, or is there a more cost-effective solution?

**R2:** We use the identifier \( z \) with effectively infinite capacity to handle the increasing number of protected entities. This approach does not require retraining the style domain encoder and incurs only minimal additional cost. We will provide more explanations.

- **First, one of the advantages of the paper is to use the _identifier \( z \)_ to address the issue of the increasing number of protected entities.** In this paper, unrestricted \( z \) can represent any text, image, video, or audio, which is encoded into \( z_{\text{emb}} \) and injected into the style domain to ensure the boundary of the protected unit.

- **Second, \( z \) signifies the identifier that maximally shifts the contraction domain to the edge distribution** of the style representation space. After decoupling the style domain and negative samples and performing dynamic contrastive learning to increase the distance in the similarity space, \( z \) is further shifted to the boundary space by injecting \( z \).

- **Third, we do not need to retrain the style domain encoder.** We only need to decouple the protected unit into the style domain, inject the corresponding identifier \( z \) to shift it to the edge distribution, and store the mapping relationship in the watermark extractor (This only incurs a minimal cost).

**Q3:** As the number of protected entities grows, will the styles of these entities influence each other, potentially reducing the model’s detection performance?

**R3:** Thank you for your comments and we do understand your concerns. In response to Question 2, **the advantages of our paper for addressing this issue are to decouple the style domain, utilize dynamic contrastive learning, and use the identifier \( z \).** To further alleviate your concerns, we provide more explanations.
- First, we decouple the latent variables into sub-vectors that control image generation, thereby extracting linearly independent combinations of the image's essential features. We utilize dynamic contrastive learning to set sample anchors, edge samples, central samples, and negative samples for the protected units, encouraging the style domain of the protected units to occupy mutually exclusive regions in the high-dimensional space.

- Second, we propose the identifier \( z \) to further enhance the reliability and security of the solution. \( z \) represents the identifier that maximally shifts the contraction domain to the edge distribution of the style representation space. Since \( z \) is an arbitrary identifier (including any text, string, image, etc.), its capacity is effectively infinite, which is sufficient to differentiate the growing number of protected entities.

- Third, we have thoroughly validated the feasibility of our approach through primary and ablation experiments. As the number of protected entities increases, the distinctiveness of the model remains robust, ensuring that the styles of these entities do not influence each other.

|           | 0-20% | 20-40% |40-60% | 60-80% | 80-90%|90-100% | TP  | TN  | Avg acc (%) ↓ | k@t@100%wd (%) ↓ |
|-----------|-------|--------|--------|--------|--------|---------|-----|-----|---------------|------------------|
| **$z$ Error**   | 0 |151| 555 |291| 3| 0 | 0 |1000| 52.15 |0|
| **$z$-watermarking** |0| 0 |0 |1| 6| **993** | 999 | 1 | **99.87** | **97.9** |

**Q4:** The annotations for these symbols and subscripts are not clear, and there is still significant room for improvement in introducing the technical flow.
**R4:** Thank you for pointing out the shortcomings in our writing. We will make revisions.


---------------------------------------------------------------------------------------------


## Reviewer jJoo

Dear Reviewer jJoo, thank you very much for your careful review of our paper and thoughtful comments. We hope the following responses can help clarify potential misunderstandings and alleviate your concerns.

**Q1:** In Table I, it is suggested to compare with some recent and SOTA watermarking methods, such as Trustmark and RoSteALS.

**R1:** Thank you for your constructive suggestions! We have added Trustmark[1] and RoSteALS[2] as comparison baselines. Here are more details and discussions: We set up 100 images with different watermarks. Trustmark has a length of 64 bits, RoSteALS has a length of 56 bits, and ours is 128 bits. We evaluated whether the generated images contained watermarks using TP (True Positive) and TN (True Negative). For watermark extraction, we used Avg acc (%) and $k@t@100\%wd$ as evaluation metrics. The experimental results indicate that previous watermarking methods are diluted or erased during the generation process, which is detrimental to the traceability and ownership of the samples.

| Method           | TP  | TN  | Avg acc (%) | $k@t@100\%wd$ (%) |
|------------------|-----|-----|-------------|-------------------|
| Trustmark | 93  | 907 | 55.37       | 6.6   |
| RoSteALS  | -   | -   | 66.50       | 7.9 |
| **Ours** | 999 | 1   | **99.83**   | **97.7**  |

[1] TrustMark: Universal Watermarking for Arbitrary Resolution Images.

[2] RoSteALS: Robust steganography using autoencoder latent space. In CVPRW 2023.

**Q2:** Can $z$-watermarking resist some watermark removal or attack methods, such as DDIM inversion or VAE? The authors could provide some results to improve the completeness of the experiment.

**R2:** Thank you for your constructive suggestions! In our paper's robustness experiments, we conducted experiments on Latent Attacks. To further demonstrate the superiority of our approach, we have included the following additional experiments. We hereby provide more details.

- First, we use the watermark removal method [1] to attack baseline watermarking schemes and ours. For attacks using variational autoencoders, we evaluate two pre-trained image compression models: Cheng2020 [2]. The compression factors are set to 3. For diffusion model attacks, we use stable diffusion 2.0 [3]. The number of noise steps is set to 60.
- Second, we chose Avg acc (average watermark accuracy), Detect Acc (percentage of images where decoded bits exceed the detection threshold 0.65), and $k@t@100\%wd$ as the evaluation metrics for watermark robustness. The result is as follows.
- Third, our method achieves an average accuracy of 97.93% and 95.81%, with a detection accuracy of 100% and $k@t@100\%wd$ of 91.5% and 87.2% under VAE and Diffusion attacks, respectively. In contrast, other methods like DCT-DWT-SVD, RivaGan, and SSL show significantly lower performance. From the results, our performance significantly surpasses other watermarking schemes after being subjected to watermark removal attacks [1].

| Method | Removal Attack Instance | Avg acc (%) | Detect Acc (%) | $k@t@100\%wd$ (%) |
|-----------|-------------------------|-------------|----------------|-------------------|
||VAE attack|50.17| 2.0|0.0|
|**DCT-DWT-SVD**|Diffusion attack| 54.41 | 2.8| 0.0|
||VAE attack|60.71|6.2| 0.0|
|**RivaGan**|Diffusion attack|58.23|1.8|0.0|
||VAE attack|62.92|15.6|0.0|
| **SSL**|Diffusion attack|63.21|16.3|0.0|
||VAE attack|**97.93**|**100** |**91.5**|
|**Ours**| Diffusion attack| 95.81|100|87.2|

[1] Zhao X, Zhang K, Su Z, et al. Invisible image watermarks are provably removable using generative ai[J]. arXiv preprint arXiv:2306.01953, 2023.

[2] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7939–7948.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

**Q3:** As the number of protected entities grows, will the styles of these entities influence each other, potentially reducing the model’s detection performance?

**R3:** Thank you for your comments and we do understand your concerns. In response to Question 2, **the advantages of our paper for addressing this issue are to decouple the style domain, utilize dynamic contrastive learning, and use the identifier $z$**. To further alleviate your concerns, we provide more explanations.

- First, we decouple the latent variables into sub-vectors that control image generation, thereby extracting linearly independent combinations of the image's essential features. We utilize dynamic contrastive learning to set sample anchors, edge samples, central samples, and negative samples for the protected units, encouraging the style domain of the protected units to occupy mutually exclusive regions in the high-dimensional space.
- Second, we propose the identifier $z$ to further enhance the reliability and security of the solution. $z$ represents the identifier that maximally shifts the contraction domain to the edge distribution of the style representation space. Since $z$ is an arbitrary identifier (including any text, string, image, etc.), its capacity is effectively infinite, which is sufficient to differentiate the growing number of protected entities.
- Third, we have thoroughly validated the feasibility of our approach through primary and ablation experiments. As the number of protected entities increases, the distinctiveness of the model remains robust, ensuring that the styles of these entities do not influence each other.

**Q4:** The annotations for these symbols and subscripts are not clear, and there is still significant room for improvement in introducing the technical flow.

**R4:** Thank you for pointing out the shortcomings in our writing. We will make revisions.


----------------------------------------------------------------------------------------------



Dear Reviewer NU2V, thank you very much for your careful review of our paper and thoughtful comments. We hope the following responses can help clarify potential misunderstandings and alleviate your concerns.

### Q1: The significant problem is the writing, too obscure.

**R1:** Thank you for your constructive suggestions! We will reorganize and improve it to make the expression clearer and more understandable.

### Q2: This paper is really hard to follow. The method's modules are not explained very well, and the input, output, and training details of each module are not clear.

**R2:** Thank you for pointing out the shortcomings in our writing description of our method. We are deeply sorry that our submission may lead you to some misunderstandings that we want to clarify here.

- We have reorganized the description of the method to explain the coherence between the method's modules, including the input, output, and training details of each module.
- **Regarding training details**, we have provided *model details and experiment details in the* *supplementary materials*. We will refine this section in the future to make it easier to follow.

### Q3: The paper also lacks simple examples of the data used in the experiment.

**R3:** Thank you for your comments and we do understand your concerns.

- **In Section 3.2 of the supplementary materials**, we generated samples using the prompt "an image of a vast grassland reminiscent of Van Gogh's 'Starry Night'" from suspicious models and APIs, including Stable Diffusion v1.5 & v2.0, PixArt-$\alpha$, PG-v2.5, DALL·E·3, and Imagen2.
- We provide strong evidence of its ability to detect imitation behavior of commercial APIs like PixArt-$\alpha$ using a few suspicious generations.

### Q4: Can you explain the meaning of each symbol of formula 7 in detail, and supplement the rationality and significance of the indicators proposed?

**R4:** Formula 7 is as follows:

$P_{z}(x|\phi \backsimeq \mathcal{D}) = \frac{q_{\phi_z}(z_{emb}|z)}{2^L \cdot K \cdot (c+\beta)^{K \times N^2 \times (K-1)}}$



- **On the left side of the equation:** \(x\) denotes the suspicious sample to be detected, \(\phi\) represents the parameters of the VAE which encodes the image into the latent space, \(\mathcal{D}\) denotes the protected dataset, \(z\) signifies the identifier that maximally shifts the contraction domain to the edge distribution of the representation space. And \(p(\cdot)\) represents the probability distribution of the copyright of \(x\) belonging to \(\mathcal{D}\).
- **On the right side of the equation:** \(z_{emb}\) denotes the embedding representation, and \(q_{\phi_z}(z_{emb}|z)\) denotes the prior probability distribution sampled in the representation space. \(L\) denotes the length of the watermark, \(K\) denotes the number of protected units, \(N\) denotes the number of generation of protected units, \(c\) denotes the marginal distance between different samples, and \(\beta\) denotes a positive hyper-parameter.

In **Eq.4** of Section 3.3 of the paper:

$\prod_{{s_k \sim \mathcal{D}_s^k}, {s_{\neg k} \sim \mathcal{D}_s^{\neg k}}} \mathbb{I}(s_k,s_{\neg k}) \gg (c+\beta)^{|\mathcal{D}_s^k|\times|\mathcal{D}_s^{\neg k}|}$

we aim to ensure the boundary of its spatial distribution so that the protection units are offset pairwise with other samples. Therefore, \(1/2^L\) denotes the probability of the watermark conforming to \(L_{bit}\), \(1/K\) denotes the probability that the sample to be detected belongs to \(K\) datasets' class, and \(1/(c+\beta)^{K\times N^2\times(K-1)}\) denotes the reciprocal of the distance between samples with different styles and contents. Their product represents the probability that the sample to be detected originates from the protected dataset. In hypothesis testing, a low-probability event is almost unlikely to occur in a single random trial, and the probability of such an event is used as the significance level \(\alpha\) (i.e., \(\alpha \leq P(\cdot)\)). Therefore, in the process of copyright ownership detection, the event that the sample is detected as belonging to the protected dataset can be expressed as \(H_0: D \leftarrow x\), with a confidence interval of \(1 - \alpha\), and can be expressed as \(P(|X - \mathcal{D}| \leq c) = 1 - \alpha\). Thus, we have a very high confidence in ensuring the accuracy of the copyright boundary and ownership.

### Q5: In 5.2 Main result, can you give the value of the number of protected units (i.e., K) and how the 1000 images used were selected?

**R5:** Thank you for your comments and we do understand your concerns.

1. In 5.2 Main result, we set the value of \(K\) to 50.
2. In the process of selecting 1000 images of the protected unit, we first obtained the representation \(z\) of each image through the style domain encoder. We randomly selected one as the \(z_o\) anchor sample, and the others as \(z_{go}\). Then, we ranked them based on their similarity and Euclidean distance, and finally selected the images according to the ranking results.

### Q6: It is observed that the avg acc is higher with the longer the watermark length in the ablation experiment. Can you give the details of the mapping from the contraction domain to the watermark in the extractor?

**R6:** Thank you for your comments and we do understand your concerns. Please note that in our ablation experiments, Acc avg shows a slight increase (±0.2) with increasing bit length. However, the other metric, \(k@t@100\%wd\), exhibits a clear downward trend. We analyze that this decline is primarily due to the model's scale law. We use PyTorch code to represent the details of the extractor, as follows:

```python
class w_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(w_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_Sequential = nn.Sequential(
            nn.Conv2d(4*in_channels, 4*in_channels, 3, 2, 1),
            nn.BatchNorm2d(4*in_channels),
            nn.GELU(),
            nn.Conv2d(4*in_channels, 4*in_channels, 3, 2, 1),
            nn.BatchNorm2d(4*in_channels),
            nn.GELU(),
        )
        self.Features_fusion = nn.Sequential(
            nn.BatchNorm1d(6*in_channels),
            nn.GELU(),
            nn.Linear(6*in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU()
        )
        self.Features_reduce = nn.Sequential(
            nn.Linear(4*in_channels, 2*in_channels),
            nn.BatchNorm1d(2*in_channels)
        )
        self.out = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.fc_d = nn.Linear(in_channels, in_channels)
        self.fc_f = nn.Linear(in_channels, in_channels)
        self.Adapt = nn.AdaptiveAvgPool2d(1)

    def forward(self, data, domain, z=None):
        f = self.Conv_Sequential(domain)
        f = self.Adapt(f).view(f.shape[0], f.shape[1])
        f_reduce = torch.cat((self.Features_reduce(f), f), dim=-1)
        f_fusion = self.Features_fusion(torch.add(f_reduce, z))
        out = self.fc_d(data) + self.fc_f(f_fusion) + data + f_fusion
        out = self.out(out)
        return out

wm_logits = Z_Model.w_decoder(data, style_and_contents_representations, z)
binary_wm = torch.round((wm_logits >= 0).float()).long()
```

### Q7: The experiment is only compared with the digital watermarking method, can you add a comparison with other methods (such as backdoor-based)?

**R7:** Thank you for this constructive suggestion! To further alleviate your concerns, we hereby compare ours and methods based on backdoor attacks.

- We employ the current state-of-the-art method, DIAGNOSIS[1], for dataset protection through backdoor techniques. It is important to note that the evaluation metrics utilized are True Positive (TP), True Negative (TN), and Attack Success Rate (ASR), as implemented by DIAGNOSIS. We use Stable Diffusion 1.5 as a surrogate model for experiments and have generated 100 images for copyright ownership verification.
- **Main result:** The experimental results are shown in the table below.

    | Method    | TP  | TN  | ASR (%) | Avg acc (%) | $k@t@100\%wd$ (%) |
    |-----------|-----|-----|---------|-------------|-------------------|
    | DIAGNOSIS | 993 | 7   | 99.3    | -           | -                 |
    | Ours      | 999 | 1   | 99.9    | 99.72       | 98                |

- **Post-tracking ownership:** Post-tracking ownership refers to the process of claiming copyright ownership when the owner discovers suspicious models or images. Due to backdoors in mimic models that have been stolen and not timely injected, effective copyright claims cannot be made.

    | Method    | TP  | TN  | ASR (%) | Avg acc (%) | $k@t@100\%wd$ (%) |
    |-----------|-----|-----|---------|-------------|-------------------|
    | DIAGNOSIS | 2   | 998 | 0.2     | -           | -                 |
    | Ours      | 999 | 1   | 99.9    | 99.69       | 94.7              |

[1] DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models. ICLR 2024.
```


--------------------------------------




Dear Reviewer NU2V, thank you very much for your careful review of our paper and thoughtful comments. We hope the following responses can help clarify potential misunderstandings and alleviate your concerns.

### Q1: The significant problem is the writing,  and This paper is really hard to follow. 
**R1:** Thank you for your constructive suggestions! We will reorganize and improve it to make the expression clearer and more understandable. **Regarding training details**, we have provided *model details and experiment details in the* *supplementary materials*. 
### Q3: The paper also lacks simple examples of the data used in the experiment.
**R3:**  **In Section 3.2 of the supplementary materials**, we generated samples using the prompt "an image of a vast grassland reminiscent of Van Gogh's 'Starry Night'" from suspicious models and APIs, including PixArt-$\alpha$, PG-v2.5, DALL·E·3, and Imagen2.
### Q4: Can you explain the meaning of each symbol of formula 7 in detail, and supplement the rationality and significance of the indicators proposed?
**R4:** Formula 7 is as follows:
$P_{z}(x|\phi \backsimeq \mathcal{D}) = \frac{q_{\phi_z}(z_{emb}|z)}{2^L \cdot K \cdot (c+\beta)^{K \times N^2 \times (K-1)}}$
- **On the left side of the equation:** $x$ denotes the suspicious sample, $\phi$ represents the parameters of the VAE , $\mathcal{D}$ denotes the protected dataset, $z$ signifies the identifier. And $p(\cdot)$ represents the probability distribution of the copyright of $x$ belonging to $\mathcal{D}$.
- **On the right side of the equation:** $z_{emb}$ denotes the embedding representation, and $q_{\phi_z}(z_{emb}|z)$ denotes the prior probability distribution. $L$ denotes the length of the watermark, $K$ denotes the number of protected units, $N$ denotes the number of generation of protected unit, $c$ denotes the marginal distance, and $\beta$ denotes a positive hyper-parameter.

In **Eq.4** of Section 3.3 of the paper, we aim to ensure the boundary of its spatial distribution so that the protection units are offset pairwise with other samples. Here, $/2^L$ denotes the probability of the watermark conforming to $L_{bit}$, $1/K$ denotes the probability that the sample to be detected belongs to $K$ datasets' class, and $1/(c+\beta)^{K\times N^2\times(K-1)}$ denotes the reciprocal of the distance between samples with different styles and contents. Their product represents the probability that the sample to be detected originates from the protected dataset. In hypothesis testing, a low-probability event is almost unlikely to occur in a single random trial, and the probability of such an event is used as the significance level $\alpha$ (i.e., $\alpha$ $ \leq P(\cdot))$. Therefore, in the process of copyright ownership detection, the event that the sample is detected as belonging to the protected dataset can be expressed as $H_0: D \leftarrow x$, with a confidence interval of $1 - \alpha\$, and can be expressed as $P(|X - \mathcal{D}| \leq c) = 1 - \alpha$. Thus, we have a very high confidence in ensuring the accuracy of the copyright boundary and ownership.
### Q5: In 5.2 Main result, can you give the value of the number of protected units (i.e., K) and how the 1000 images used were selected?
**R5:** Thank you for your comments and we do understand your concerns.
1. In 5.2 Main result, we set the value of $K$ to 50.
2. In the process of selecting 1000 images of the protected unit, we first obtained the representation $z$ of each image through the style domain encoder. We randomly selected one as the $z_o$ anchor sample, and the others as $z_{go}$. Then, we ranked them based on their similarity and Euclidean distance, and finally selected the images according to the ranking results.

### Q6: It's observed that the avg acc is higher with the longer the watermark length. Can you give the details of the mapping from the contraction domain to the watermark in the extractor?
**R6:** Thank you for your comments. Please note that in our ablation experiments, Acc avg shows a slight increase (±0.2) with increasing bit length. However, the other metric, $k@t@100\%wd$, exhibits a downward trend. We analyze that this decline is primarily due to the model's scale law. PyTorch code as follows:
```
class w_decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(w_decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv_Sequential = nn.Sequential(
            nn.Conv2d(4*in_channels, 4*in_channels, 3, 2, 1),
            nn.BatchNorm2d(4*in_channels),
            nn.GELU(),
            nn.Conv2d(4*in_channels, 4*in_channels, 3, 2, 1),
            nn.BatchNorm2d(4*in_channels),
            nn.GELU(),)
        self.Features_fusion = nn.Sequential(
            nn.BatchNorm1d(6*in_channels),
            nn.GELU(),
            nn.Linear(6*in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU())
        self.Features_reduce = nn.Sequential(
            nn.Linear(4*in_channels, 2*in_channels),
            nn.BatchNorm1d(2*in_channels))
        self.out = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels))
        self.fc_d = nn.Linear(in_channels, in_channels)
        self.fc_f = nn.Linear(in_channels, in_channels)
        self.Adapt = nn.AdaptiveAvgPool2d(1)
    def forward(self, data, domain, z=None):
        f = self.Conv_Sequential(domain)
        f = self.Adapt(f).view(f.shape[0], f.shape[1])
        f_reduce = torch.cat((self.Features_reduce(f), f), dim=-1)
        f_fusion = self.Features_fusion(torch.add(f_reduce, z))
        out = self.fc_d(data) + self.fc_f(f_fusion) + data + f_fusion
        out = self.out(out)
        return out
wm_logits = Z_Model.w_decoder(data, domain, z)
```
### Q7: The experiment is only compared with the digital watermarking method, can you add a comparison with other methods (such as backdoor-based)?

**R7:** To further alleviate your concerns, we compare ours and methods based on backdoor attacks.

- We employ the current state-of-the-art method, DIAGNOSIS[1], for dataset protection through backdoor. It's important to note that the evaluation metrics utilized are True Positive (TP), True Negative (TN), and Attack Success Rate (ASR), as implemented by DIAGNOSIS. 
- **Main result:** The experimental results are shown in the table below.

    | Method    | TP  | TN  | ASR (%) | Avg acc (%) | $k@t@100\%wd$ (%) |
    |-----------|-----|-----|---------|-------------|-------------------|
    | DIAGNOSIS | 993 | 7   | 99.3    | -           | -                 |
    | Ours      | 999 | 1   | 99.9    | 99.72       | 98                |

- **Post-tracking ownership:** Post-tracking ownership refers to the process of claiming copyright ownership when the owner discovers suspicious models or images. Due to backdoors in mimic models that have been stolen and not timely injected, effective copyright claims cannot be made.

    | Method    | TP  | TN  | ASR (%) | Avg acc (%) | $k@t@100\%wd$ (%) |
    |-----------|-----|-----|---------|-------------|-------------------|
    | DIAGNOSIS | 2   | 998 | 0.2     | -           | -                 |
    | Ours      | 999 | 1   | 99.9    | 99.69       | 94.7              |

[1] DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models. ICLR 2024.
