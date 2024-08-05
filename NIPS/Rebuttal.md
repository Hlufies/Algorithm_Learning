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
| Trustmark        | 93  | 907 | 55.37       | 6.6               |
| RoSteALS         | -   | -   | 66.50       | 7.9               |
| **Ours**         | 999 | 1   | **99.83**   | **97.7**          |

[1] TrustMark: Universal Watermarking for Arbitrary Resolution Images.

[2] RoSteALS: Robust steganography using autoencoder latent space. In CVPRW 2023.

**Q2:** Can $z$-watermarking resist some watermark removal or attack methods, such as DDIM inversion or VAE? The authors could provide some results to improve the completeness of the experiment.

**R2:** Thank you for your constructive suggestions! We agree that understanding the impact of watermark removal is also important. In our paper's robustness experiments, we conducted experiments on Latent Attacks. To further demonstrate the superiority of our approach, we have included the following additional experiments. We hereby provide more details and discussions.

- First, we use the watermark removal method [1] to attack baseline watermarking schemes and ours. For attacks using variational autoencoders, we evaluate two pre-trained image compression models: Cheng2020 [2]. The compression factors are set to 3. For diffusion model attacks, we use stable diffusion 2.0 [3]. The number of noise steps is set to 60.
- Second, we chose Avg acc (average watermark accuracy), Detect Acc (percentage of images where decoded bits exceed the detection threshold 0.65), and $k@t@100\%wd$ as the evaluation metrics for watermark robustness. The result is as follows.
- Third, our method achieves an average accuracy of 97.93% and 95.81%, with a detection accuracy of 100% and $k@t@100\%wd$ of 91.5% and 87.2% under VAE and Diffusion attacks, respectively. In contrast, other methods like DCT-DWT-SVD, RivaGan, and SSL show significantly lower performance. From the results, our performance significantly surpasses other watermarking schemes after being subjected to watermark removal attacks [1].

| Method    | Removal Attack Instance | Avg acc (%) | Detect Acc (%) | $k@t@100\%wd$ (%) |
|-----------|-------------------------|-------------|----------------|-------------------|
|           | VAE attack              | 50.17       | 2.0            | 0.0               |
| **DCT-DWT-SVD** | Diffusion attack        | 54.41       | 2.8            | 0.0               |
|           | VAE attack              | 60.71       | 6.2            | 0.0               |
| **RivaGan**    | Diffusion attack        | 58.23       | 1.8            | 0.0               |
|           | VAE attack              | 62.92       | 15.6           | 0.0               |
| **SSL**        | Diffusion attack        | 63.21       | 16.3           | 0.0               |
|           | VAE attack              | **97.93**   | **100**        | **91.5**          |
| **Ours**       | Diffusion attack        | 95.81       | 100            | 87.2              |

[1] Zhao X, Zhang K, Su Z, et al. Invisible image watermarks are provably removable using generative ai[J]. arXiv preprint arXiv:2306.01953, 2023.

[2] Z. Cheng, H. Sun, M. Takeuchi, and J. Katto, “Learned image compression with discretized gaussian mixture likelihoods and attention modules,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 7939–7948.

[3] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 684–10 695.

**Q3:** As the number of protected entities grows, will the styles of these entities influence each other, potentially reducing the model’s detection performance?

**R3:** Thank you for your comments and we do understand your concerns. In response to Question 2, **the advantages of our paper for addressing this issue are to decouple the style domain, utilize dynamic contrastive learning, and use the identifier $z$**. To further alleviate your concerns, we provide more explanations.

- First, we decouple the latent variables into sub-vectors that control image generation, thereby extracting linearly independent combinations of the image's essential features. We utilize dynamic contrastive learning to set sample anchors, edge samples, central samples, and negative samples for the protected units, encouraging the style domain of the protected units to occupy mutually exclusive regions in the high-dimensional space.
- Second, we propose the identifier $z$ to further enhance the reliability and security of the solution. $z$ represents the identifier that maximally shifts the contraction domain to the edge distribution of the style representation space. Since $z$ is an arbitrary identifier (including any text, string, image, etc.), its capacity is effectively infinite, which is sufficient to differentiate the growing number of protected entities.
- Third, we have thoroughly validated the feasibility of our approach through primary and ablation experiments. As the number of protected entities increases, the distinctiveness of the model remains robust, ensuring that the styles of these entities do not influence each other.

## Table

| Method    | Removal Attack Instance | Avg acc (%) | Detect Acc (%) | $k@t@100\%wd$ (%) |
|-----------|-------------------------|-------------|----------------|-------------------|
|           | VAE attack              | 50.17       | 2.0            | 0.0               |
| **DCT-DWT-SVD** | Diffusion attack        | 54.41       | 2.8            | 0.0               |
|           | VAE attack              | 60.71       | 6.2            | 0.0               |
| **RivaGan**    | Diffusion attack        | 58.23       | 1.8            | 0.0               |
|           | VAE attack              | 62.92       | 15.6           | 0.0               |
| **SSL**        | Diffusion attack        | 63.21       | 16.3           | 0.0               |
|           | VAE attack              | **97.93**   | **100**        | **91.5**          |
| **Ours**       | Diffusion attack        | 95.81       | 100            | 87.2              |

**Q4:** The annotations for these symbols and subscripts are not clear, and there is still significant room for improvement in introducing the technical flow.

**R4:** Thank you for pointing out the shortcomings in our writing. We will make revisions.

