# 实验归档与服务器迁移（2026-11-18）

## 文档定位

| 用途 | 路径 |
| --- | --- |
| 终版实验档案 | `docs/classmean_procrustes_training.md` |
| 完整调查总述 | `docs/investigation_summary.md` |
| 最终 checkpoint | `exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v10/model/epoch_9.pth` |
| 最终运行配置 | `exp/smoke-ptv3m1-16-scannetpp-v2-classprocrustes-v10/config.py` |

路径以服务器 `/home/isom/cyf/CompressedSceneSplat` 为根。以上定位依据已提交档案；服务器现存文件由现场盘点确认。

档案记录：v10-E9，50 个验证场景，前景 mIoU 24.07%，全类 mIoU 24.89%，使用每场景 oracle Procrustes Q。该评测条件是结果适用范围的一部分。仓库的 `configs/custom/lang-pretrain-ptv3m1-scannetpp-v2-classprocrustes.py` 已指向 v26，复现 v10 使用实验目录的配置转储。

## 下载与验收顺序

1. 盘点服务器 Git 状态、目录容量、checkpoint、归档文件、链接与运行进程。
2. 保存 v10 的 checkpoint、运行配置、训练及评测日志、结果文件和实际调用命令。复用现存归档；同时保存服务器侧代码修改与其他有保留价值的实验记录。
3. 根据配置登记外部数据、压缩特征、SVD 基、符号注册表、文本嵌入与初始化权重的实际路径；为项目目录内的依赖安排下载。
4. 在本地下载到独立归档目录，保存服务器相对路径与 SHA-256 清单。
5. 校验下载文件数量、大小、SHA-256，并核对 checkpoint 对应的配置和评测证据。校验过程使用 CPU 文件读取。
6. 依据用户选定的目录清单清理服务器，核验释放容量和保留路径。

执行责任遵循根目录 `AGENTS.md`：服务器命令由用户手动运行；每一步根据返回结果生成下一步命令。清理阶段以前述验收完成为前提。

## 本地 LitePT 修改备份

`docs/patches/LitePT-local-pointrope.patch` 保存本次提交时的本地子模块修改，基于 LitePT commit `079709859f02460a5ddbb9ef38f2ba01eabc9f3e`。

恢复方式：在该 LitePT 版本上运行 `git -C LitePT apply ../docs/patches/LitePT-local-pointrope.patch`。

该补丁包含 CPython 3.11 扩展文件名及张量布局修改；本次仅保存原有修改，运行效果以目标环境实测为准。
