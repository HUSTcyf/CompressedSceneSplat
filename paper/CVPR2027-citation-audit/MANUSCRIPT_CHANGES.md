# 参考文献核对涉及的正文修改

对照版本为 Windows Oleafly 的 2026-11-17 11:50:50 检查点 `1c4c27074f44`，完整源文件保存在 `before-citation-audit/`。当前版本以该检查点为基准，保留原有章节结构。`main.tex` 从 `\section{Method}` 开始至结尾与检查点逐字一致，包含方法、实验、结论和 LLM 使用声明。参考文献源条目保留下载后的内容。

## 保留的必要修正

- SceneSplat 高斯数量：按 [SceneSplat 项目](https://unique1i.github.io/SceneSplat_webpage/)列出的 7K 数据集 112.7 亿个高斯，将笼统的“数百亿”调整为“数十亿”。
- CLIP、DINO 与 LERF：CLIP 指向[原论文](https://proceedings.mlr.press/v139/radford21a.html)，视觉表征 DINO 指向 [Caron 等的自监督视觉 Transformer 论文](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)，[LERF-TOGO](https://proceedings.mlr.press/v229/rashid23a.html) 标为 LERF 的抓取应用。
- 单场景方法与语义分组：OpenScene 从 3DGS 单场景优化方法的引用组移出，InstanceGaussian 从语言码本方法引用组移出，GOI 从语义聚类引用组移出。[OpenGaussian](https://3d-aigc.github.io/OpenGaussian/) 的两级码本用于离散化实例特征，[InstanceGaussian](https://openaccess.thecvf.com/content/CVPR2025/html/Li_InstanceGaussian_Appearance-Semantic_Joint_Gaussian_Representation_for_3D_Instance-Level_Perception_CVPR_2025_paper.html) 联合学习外观与语义并聚合实例，[VoteSplat](https://openaccess.thecvf.com/content/ICCV2025/papers/Jiang_VoteSplat_Hough_Voting_Gaussian_Splatting_for_3D_Scene_Understanding_ICCV_2025_paper.pdf) 使用 Hough 投票。
- 免语义优化方法：[Occam’s LGS](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_694/paper.pdf) 使用带权特征聚合，[LUDVIG](https://openaccess.thecvf.com/content/ICCV2025/papers/Marrie_LUDVIG_Learning-Free_Uplifting_of_2D_Visual_Features_to_Gaussian_Splatting_ICCV_2025_paper.pdf) 利用 3D 几何进行图扩散，[VALA](https://vala3d.github.io/) 使用可见性门控。正文按这些方法各自的处理方式表述。
- 模型压缩：[NSVD](https://arxiv.org/abs/2503.17101) 是激活感知分解，[UniQL](https://proceedings.iclr.cc/paper_files/paper/2026/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html) 结合量化和低秩压缩，[DeltaLLM](https://arxiv.org/abs/2501.18596) 结合跨层权重共享和低秩差分。[LangSplat](https://openaccess.thecvf.com/content/CVPR2024/papers/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.pdf) 提供场景级自编码器降维的引用，[Dr. Splat](https://openaccess.thecvf.com/content/CVPR2025/papers/Jun-Seong_Dr._Splat_Directly_Referring_3D_Gaussian_Splatting_via_Direct_Language_CVPR_2025_paper.pdf) 使用乘积量化。
- 附录：实矩阵 SVD 的转置写法适用于实数域。自编码器段落按 LangSplat 的自编码器、[LangSplatV2](https://arxiv.org/abs/2507.07136) 的稀疏系数以及 Occam’s LGS 的直接聚合分别归类。

## 篇幅与构建

正文结束于第 8 页。以 PDF 前 8 页的可见含英文字母词项粗略统计，检查点版本为 3,866，当前版本为 3,836，约相差 0.8%。该统计包含图表标题等可见内容，用于比较篇幅。
