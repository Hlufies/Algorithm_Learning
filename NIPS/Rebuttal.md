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
