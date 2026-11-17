# LangZip 参考文献逐条核对

## 本轮原始 BibTeX 复制说明

最终 67 条引用中，66 条直接复制自本目录的 arXiv、会议、出版社或项目官方 BibTeX，字段内容保持来源原样，仅将引用键改为论文现有键。逐条来源、原始引用键和 SHA-256 见 `source-copy-manifest.json`。第 67 条 `vonNeumann1937` 沿用已有条目，元数据与 Netlib 的冯·诺依曼文献目录核对一致。ARKitScenes 采用 Apple 项目提供的 NeurIPS 原始条目，Hypersim 和 RIO 采用 CVF 官方条目。CVF 的 RIO BibTeX 没有页码字段，本次按原文复制。原始条目的版本年份和正式会议发表年份可能不同，以实际采用的条目为准。

为使原始 BibTeX 中的 Unicode 标点和月份写法正确排版，LaTeX 源文件增加标点映射，参考文献文件增加月份宏定义。`ieeenat_fullname.bst` 与 `cvpr.sty` 均与 CVPR 2027 官方 CVPR2027-v1 版本逐字节一致。条目字段保持原样。以下逐条核对记录保留了上一轮的版本说明与资料链接，实际采用的版本以 `source-copy-manifest.json` 和各 `.bib` 文件为准。

## 核对范围

- 原始正文和附录实际引用 69 个 BibTeX 键。核对后合并重复的 CLIP 条目并将 Big Vision 的错误引用改为 CLIP 原论文，最终保留 67 条。
- 表格逐条给出最终来源及 arXiv 链接。`arxiv-bibtex/` 保存 arXiv 原始 BibTeX，`cvf-bibtex/`、`venue-bibtex/` 和 `publisher-bibtex/` 保存会议或出版社原始记录。
- 最终文献中 59 条记录具有已核验的 arXiv 编号。经典期刊文献、仅收录于会议的论文及代码仓库使用原始出版社、会议或项目页面。
- 正式会议 PDF 与网站元数据出现差异时，作者名单以会议论文首页和 arXiv 交叉核对，差异记录如下。

## 重点修正

- ARKitScenes：改为 Gilad Baruch 等 11 位作者。[arXiv](https://arxiv.org/abs/2111.08897)；[项目官方引用](https://github.com/apple/ARKitScenes)。
- Hypersim：修正作者名单，并采用 CVF ICCV 官方条目的页码 10912–10922。检查报告中的 10952–10962 与 CVF 条目不同。[CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Roberts_Hypersim_A_Photorealistic_Synthetic_Dataset_for_Holistic_Indoor_Scene_Understanding_ICCV_2021_paper.html)。
- 3RScan：改为介绍该数据集的 RIO 论文及其 7658–7667 页码。[ICCV 论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.pdf)。
- ADMM 综述：补入首位作者 Stephen Boyd，并修正其他作者姓名。[作者网页](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html)。
- NSVD：原条目为张量分解中的同名方法，与正文描述的激活感知大语言模型压缩方法不同。正文现引用 [Lu 等人的 NSVD](https://arxiv.org/abs/2503.17101)。
- DINO：原条目是目标检测方法 DETR with Improved DeNoising Anchor Boxes，正文讨论的是视觉表征模型。现引用 [Caron 等人的 DINO](https://arxiv.org/abs/2104.14294)。
- DINOv2 作者顺序以及 Replica 数据集作者姓名已依据 arXiv 记录更正。
- Occam’s LGS 使用正式 BMVC 2025 题名 *An Efficient Approach for Language Gaussian Splatting*。
- SceneSplat++：NeurIPS 网页及 BibTeX 列出 14 位作者，但[会议论文首页](https://proceedings.neurips.cc/paper_files/paper/2025/file/6b9628a5da76f5f1d6ec025f4b493686-Paper-Datasets_and_Benchmarks_Track.pdf)及 [arXiv](https://arxiv.org/abs/2506.08710) 均列出 13 位。当前条目按论文首页及 arXiv 的 13 位作者填写，建议由论文作者向该文作者核对会议元数据差异。
- OpenNeRF：ICLR 网站 BibTeX 列出 5 位作者，但[会议论文首页](https://proceedings.iclr.cc/paper_files/paper/2024/file/21f1c5bbf2519321c1bee9bfa9edcd46-Paper-Conference.pdf)及 [arXiv](https://arxiv.org/abs/2404.03650) 均列出 6 位。当前条目按论文首页及 arXiv 填写 6 位。
- Dobi-SVD：采用 ICLR 2025 论文首页的 5 位作者。该正式版本与 arXiv 更新版本的作者列表不同。

## 原始 69 个引用键逐条记录

| 原引用键 | 核对后题名或处理 | 来源 | arXiv |
|---|---|---|---|
| `big_vision` | CLIP 引用改为原论文，基础模型列表移除代码库引用 | [CLIP 正式论文](https://proceedings.mlr.press/v139/radford21a.html) / [原代码库](https://github.com/google-research/big_vision) | [2103.00020](https://arxiv.org/abs/2103.00020) |
| `cai2010singular` | A singular value thresholding algorithm for matrix completion | [来源](https://doi.org/10.1137/080738970) | [ 0810.3286 ](https://arxiv.org/abs/0810.3286) |
| `RPCA` | Robust principal component analysis? | [来源](https://doi.org/10.1145/1970392.1970395) | [ 0912.3599 ](https://arxiv.org/abs/0912.3599) |
| `chang2017matterport3d` | Matterport3D: Learning from RGB-D Data in Indoor Environments | [来源](https://doi.org/10.1109/3dv.2017.00081) | [ 1709.06158 ](https://arxiv.org/abs/1709.06158) |
| `cheng2024occamslgssimpleapproach` | Occam’s LGS: An Efficient Approach for Language Gaussian Splatting | [来源](https://bmvc2025.bmva.org/proceedings/694/) | [ 2412.01807 ](https://arxiv.org/abs/2412.01807) |
| `chiang2025uniql` | UniQL: Unified Quantization and Low-rank Compression for Adaptive Edge LLMs | [来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/370fa2e691f57eb319bc263a07dad4a5-Abstract-Conference.html) | [ 2512.03383 ](https://arxiv.org/abs/2512.03383) |
| `pointcept2023` | Pointcept: A Codebase for Point Cloud Perception Research | [来源](https://github.com/Pointcept/Pointcept) | — |
| `dai2017scannet` | ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes | [来源](https://openaccess.thecvf.com/CVPR2017.py?day=all) | [ 1702.04405 ](https://arxiv.org/abs/1702.04405) |
| `dehghan2021arkitscenes` | ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene Understanding Using Mobile RGB-D Data | [来源](https://github.com/apple/ARKitScenes) | [ 2111.08897 ](https://arxiv.org/abs/2111.08897) |
| `ding2022language` | PLA: Language-Driven Open-Vocabulary 3D Scene Understanding | [来源](https://openaccess.thecvf.com/CVPR2023?day=all) | [ 2211.16312 ](https://arxiv.org/abs/2211.16312) |
| `engelmann2024opennerf` | OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views | [来源](https://proceedings.iclr.cc/paper_files/paper/2024/file/21f1c5bbf2519321c1bee9bfa9edcd46-Paper-Conference.pdf) | [ 2404.03650 ](https://arxiv.org/abs/2404.03650) |
| `SVD` | Calculating the Singular Values and Pseudo-Inverse of a Matrix | [来源](https://doi.org/10.1137/0702016) | — |
| `guo2024semantic` | Semantic Gaussians: Open-Vocabulary Scene Understanding with 3D Gaussian Splatting | [来源](https://arxiv.org/abs/2403.15624) | [ 2403.15624 ](https://arxiv.org/abs/2403.15624) |
| `gupta2015indoor` | Indoor Scene Understanding with RGB-D Images: Bottom-up Segmentation, Object Detection and Semantic Segmentation | [来源](https://s-gupta.github.io/) | — |
| `jiang2024open` | Open-Vocabulary 3D Semantic Segmentation with Foundation Models | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | — |
| `jiang2025votesplat` | VoteSplat: Hough Voting Gaussian Splatting for 3D Scene Understanding | [来源](https://openaccess.thecvf.com/ICCV2025?day=all) | [ 2506.22799 ](https://arxiv.org/abs/2506.22799) |
| `drsplat25` | Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration | [来源](https://openaccess.thecvf.com/CVPR2025?day=all) | [ 2502.16652 ](https://arxiv.org/abs/2502.16652) |
| `kerbl3Dgaussians` | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | [来源](https://arxiv.org/abs/2308.04079) | [ 2308.04079 ](https://arxiv.org/abs/2308.04079) |
| `lerf2023` | LERF: Language Embedded Radiance Fields | [来源](https://openaccess.thecvf.com/ICCV2023?day=all) | [ 2303.09553 ](https://arxiv.org/abs/2303.09553) |
| `kirillov2023segany` | Segment Anything | [来源](https://openaccess.thecvf.com/ICCV2023?day=all) | [ 2304.02643 ](https://arxiv.org/abs/2304.02643) |
| `lee2025mosaic3d` | Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation | [来源](https://openaccess.thecvf.com/CVPR2025?day=all) | [ 2502.02548 ](https://arxiv.org/abs/2502.02548) |
| `li2025instancegaussian` | InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception | [来源](https://openaccess.thecvf.com/CVPR2025?day=all) | [ 2411.19235 ](https://arxiv.org/abs/2411.19235) |
| `li2025langsplatv2` | LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS | [来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/feb14cd8a37e800f38edba497e847363-Abstract-Conference.html) | [ 2507.07136 ](https://arxiv.org/abs/2507.07136) |
| `li2025scenesplat` | SceneSplat: Gaussian Splatting-based Scene Understanding with Vision-Language Pretraining | [来源](https://openaccess.thecvf.com/ICCV2025?day=all) | [ 2503.18052 ](https://arxiv.org/abs/2503.18052) |
| `liang2024supergseg` | SuperGSeg: Open-Vocabulary 3D Segmentation with Structured Super-Gaussians | [来源](https://arxiv.org/abs/2412.10231) | [ 2412.10231 ](https://arxiv.org/abs/2412.10231) |
| `liu2023weakly` | Weakly Supervised 3D Open-vocabulary Segmentation | [来源](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76b693f36916a5ed84d6e5b39a0dc03-Abstract-Conference.html) | [ 2305.14093 ](https://arxiv.org/abs/2305.14093) |
| `ma2025scenesplatpp` | SceneSplat++: A Large Dataset and Comprehensive Benchmark for Language Gaussian Splatting | [来源](https://proceedings.neurips.cc/paper_files/paper/2025/file/6b9628a5da76f5f1d6ec025f4b493686-Paper-Datasets_and_Benchmarks_Track.pdf) | [ 2506.08710 ](https://arxiv.org/abs/2506.08710) |
| `marrie2025ludvig` | LUDVIG: Learning-Free Uplifting of 2D Visual Features to Gaussian Splatting Scenes | [来源](https://openaccess.thecvf.com/ICCV2025?day=all) | [ 2410.14462 ](https://arxiv.org/abs/2410.14462) |
| `mikaelyan2025deltallm` | DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights | [来源](https://arxiv.org/abs/2501.18596) | [ 2501.18596 ](https://arxiv.org/abs/2501.18596) |
| `mildenhall2020nerf` | NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis | [来源](https://arxiv.org/abs/2003.08934) | [ 2003.08934 ](https://arxiv.org/abs/2003.08934) |
| `neal2011distributed` | Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers | [来源](https://web.stanford.edu/~boyd/papers/admm_distr_stats.html) | — |
| `oquab2023dinov2` | DINOv2: Learning Robust Visual Features without Supervision | [来源](https://arxiv.org/abs/2304.07193) | [ 2304.07193 ](https://arxiv.org/abs/2304.07193) |
| `Proximal` | Proximal Algorithms | [来源](https://doi.org/10.1561/2400000003) | — |
| `peng2023openscene` | OpenScene: 3D Scene Understanding With Open Vocabularies | [来源](https://openaccess.thecvf.com/CVPR2023?day=all) | [ 2211.15654 ](https://arxiv.org/abs/2211.15654) |
| `piekenbrinck2025opensplat3d` | OpenSplat3D: Open-Vocabulary 3D Instance Segmentation using Gaussian Splatting | [来源](https://doi.org/10.1109/cvprw67362.2025.00520) | [ 2506.07697 ](https://arxiv.org/abs/2506.07697) |
| `qin2024langsplat` | LangSplat: 3D Language Gaussian Splatting | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2312.16084 ](https://arxiv.org/abs/2312.16084) |
| `qu2024goi` | GOI: Find 3D Gaussians of Interest with an Optimizable Open-vocabulary Semantic-space Hyperplane | [来源](https://doi.org/10.1145/3664647.3680852) | [ 2405.17596 ](https://arxiv.org/abs/2405.17596) |
| `DBLP:journals/corr/abs-2103-00020` | 并入 CLIP 论文引用 | [CLIP 正式论文](https://proceedings.mlr.press/v139/radford21a.html) | [2103.00020](https://arxiv.org/abs/2103.00020) |
| `radford2021learning` | Learning Transferable Visual Models From Natural Language Supervision | [来源](https://proceedings.mlr.press/v139/radford21a.html) | [ 2103.00020 ](https://arxiv.org/abs/2103.00020) |
| `rashid2023language` | Language Embedded Radiance Fields for Zero-Shot Task-Oriented Grasping | [来源](https://proceedings.mlr.press/v229/rashid23a.html) | [ 2309.07970 ](https://arxiv.org/abs/2309.07970) |
| `ravi2024sam2` | SAM 2: Segment Anything in Images and Videos | [来源](https://arxiv.org/abs/2408.00714) | [ 2408.00714 ](https://arxiv.org/abs/2408.00714) |
| `roberts2021hypersim` | Hypersim: A Photorealistic Synthetic Dataset for Holistic Indoor Scene Understanding | [来源](https://openaccess.thecvf.com/ICCV2021?day=all) | [ 2011.02523 ](https://arxiv.org/abs/2011.02523) |
| `rosinol20203d` | 3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects, and Humans | [来源](https://doi.org/10.15607/rss.2020.xvi.079) | [ 2002.06289 ](https://arxiv.org/abs/2002.06289) |
| `shi2024language` | Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2311.18482 ](https://arxiv.org/abs/2311.18482) |
| `straub2019replica` | The Replica Dataset: A Digital Replica of Indoor Spaces | [来源](https://arxiv.org/abs/1906.05797) | [ 1906.05797 ](https://arxiv.org/abs/1906.05797) |
| `tian2025ccl` | CCL-LGS: Contrastive Codebook Learning for 3D Language Gaussian Splatting | [来源](https://openaccess.thecvf.com/ICCV2025?day=all) | [ 2505.20469 ](https://arxiv.org/abs/2505.20469) |
| `tian2024drivevlm` | DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models | [来源](https://arxiv.org/abs/2402.12289) | [ 2402.12289 ](https://arxiv.org/abs/2402.12289) |
| `tschannen2025siglip` | SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features | [来源](https://arxiv.org/abs/2502.14786) | [ 2502.14786 ](https://arxiv.org/abs/2502.14786) |
| `tsitsikas2020nsvd` | Large Language Model Compression via the Nested Activation-Aware Decomposition（替换误引） | [来源](https://arxiv.org/abs/2503.17101) | [ 2503.17101 ](https://arxiv.org/abs/2503.17101) |
| `vonNeumann1937` | Some matrix-inequalities and metrization of matrix-space | [来源](https://web.math.princeton.edu/~naor/homepage%20files/nuclear-dimension.pdf) | — |
| `wald20193rscan` | RIO: 3D Object Instance Re-Localization in Changing Indoor Environments | [来源](https://openaccess.thecvf.com/content_ICCV_2019/html/Wald_RIO_3D_Object_Instance_Re-Localization_in_Changing_Indoor_Environments_ICCV_2019_paper.html) | [ 1908.06109 ](https://arxiv.org/abs/1908.06109) |
| `wang2025dobi` | Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives | [来源](https://proceedings.iclr.cc/paper_files/paper/2025/hash/218ca0d92e6ed8f9db00621e103dc70c-Abstract-Conference.html) | [ 2502.02723 ](https://arxiv.org/abs/2502.02723) |
| `wang2025vala` | Visibility-Aware Language Aggregation for Open-Vocabulary Segmentation in 3D Gaussian Splatting | [来源](https://arxiv.org/abs/2509.05515) | [ 2509.05515 ](https://arxiv.org/abs/2509.05515) |
| `wang2025qsvd` | QSVD: Efficient Low-rank Approximation for Unified Query-Key-Value Weight Compression in Low-Precision Vision-Language Models | [来源](https://proceedings.neurips.cc/paper_files/paper/2025/hash/028ef7e68a5ea25fc26cd6abf3a5c147-Abstract-Conference.html) | [ 2510.16292 ](https://arxiv.org/abs/2510.16292) |
| `wu2022ptv2` | Point Transformer V2: Grouped Vector Attention and Partition-based Pooling | [来源](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html) | [ 2210.05666 ](https://arxiv.org/abs/2210.05666) |
| `wu2024ptv3` | Point Transformer V3: Simpler Faster Stronger | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2312.10035 ](https://arxiv.org/abs/2312.10035) |
| `wu2024ppt` | Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2308.09718 ](https://arxiv.org/abs/2308.09718) |
| `wu2024opengaussian` | OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding | [来源](https://proceedings.neurips.cc/paper_files/paper/2024/hash/21f7b745f73ce0d1f9bcea7f40b1388e-Abstract-Conference.html) | [ 2406.02058 ](https://arxiv.org/abs/2406.02058) |
| `yang2024regionplc` | RegionPLC: Regional Point-Language Contrastive Learning for Open-World 3D Scene Understanding | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2304.00962 ](https://arxiv.org/abs/2304.00962) |
| `ye2024gaussian` | Gaussian Grouping: Segment and Edit Anything in 3D Scenes | [来源](https://doi.org/10.1007/978-3-031-73397-0_10) | [ 2312.00732 ](https://arxiv.org/abs/2312.00732) |
| `yeshwanth2023scannet++` | ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes | [来源](https://openaccess.thecvf.com/ICCV2023?day=all) | [ 2308.11417 ](https://arxiv.org/abs/2308.11417) |
| `yu2023compressing` | Compressing Transformers: Features Are Low-Rank, but Weights Are Not! | [来源](https://ojs.aaai.org/index.php/AAAI/article/view/26304) | — |
| `yuelitept2026` | LitePT: Lighter Yet Stronger Point Transformer | [来源](https://openaccess.thecvf.com/CVPR2026?day=all) | [ 2512.13689 ](https://arxiv.org/abs/2512.13689) |
| `zhai2023sigmoid` | Sigmoid Loss for Language Image Pre-Training | [来源](https://openaccess.thecvf.com/ICCV2023?day=all) | [ 2303.15343 ](https://arxiv.org/abs/2303.15343) |
| `zhan2025hi` | Hi-LSplat: Hierarchical 3D Language Gaussian Splatting | [来源](https://arxiv.org/abs/2506.06822) | [ 2506.06822 ](https://arxiv.org/abs/2506.06822) |
| `zhang2022dino` | Emerging Properties in Self-Supervised Vision Transformers（替换误引） | [来源](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html) | [ 2104.14294 ](https://arxiv.org/abs/2104.14294) |
| `zhou2020cylinder3d` | Cylinder3D: An Effective 3D Framework for Driving-scene LiDAR Semantic Segmentation | [来源](https://arxiv.org/abs/2008.01550) | [ 2008.01550 ](https://arxiv.org/abs/2008.01550) |
| `zhou2024feature` | Feature 3DGS: Supercharging 3D Gaussian Splatting to Enable Distilled Feature Fields | [来源](https://openaccess.thecvf.com/CVPR2024?day=all) | [ 2312.03203 ](https://arxiv.org/abs/2312.03203) |
| `zuo2025fmgs` | FMGS: Foundation Model Embedded 3D Gaussian Splatting for Holistic 3D Scene Understanding | [来源](https://doi.org/10.1007/s11263-024-02183-8) | [ 2401.01970 ](https://arxiv.org/abs/2401.01970) |

## 引用语义调整

- CLIP、DINO 和 NSVD 的正文引用分别指向与叙述内容对应的论文。
- 修正 Occam’s LGS、LangSplatV2、OpenScene、InstanceGaussian、VoteSplat、LUDVIG 与 DeltaLLM 的方法归类或描述。
- SceneSplat 相关数据规模描述改为与其数据集论文一致的数量级。
- 公开的第三方 Pointcept 仓库保留为代码库引用。
