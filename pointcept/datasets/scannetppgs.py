import os
import numpy as np

from pointcept.utils.cache import shared_dict
from pointcept.utils.svd_sign import canonicalize_svd_sign, remove_scene_mean

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class ScanNetPPGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment",
        "segment200",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "valid_feat_mask",
        "normal",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment", "pc_instance"]

    # denoting tailed classes, below 0.01% of total vertices
    TAIL_CLASSES = np.array(
        [
            82,
            74,
            84,
            76,
            86,
            80,
            85,
            92,
            88,
            83,
            93,
            87,
            94,
            89,
            55,
            90,
            96,
            97,
            91,
            95,
            98,
        ]
    )

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        load_compressed_lang_feat=False,
        svd_rank=16,
        svd_center=False,
        global_sign_path=None,
        target_rotate_matrix_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train
        self.load_compressed_lang_feat = load_compressed_lang_feat
        self.svd_rank = svd_rank
        self.svd_center = svd_center
        # 2026-08-03 全局符号对齐注册表（只读 sidecar，原始 npz 不动）：
        # {chunk_basename: signs[16]}，加载时对压缩特征按维翻转符号。
        # 生成脚本：tools/compression/build_global_sign_registry.py
        self.global_signs = None
        if global_sign_path and os.path.exists(global_sign_path):
            reg = np.load(global_sign_path, allow_pickle=True)
            self.global_signs = {
                str(n): s for n, s in zip(reg["names"], reg["signs"])
            }
            print(
                f"current: loaded global sign registry ({len(self.global_signs)} chunks) "
                f"from {global_sign_path}"
            )
        # 2026-08-04 训练目标正交旋转（rebuttal Table 2 ③）：固定随机正交矩阵，
        # canonicalize/global_signs 之后应用，把目标整体转到"等价基"。只影响训练
        # 目标（val/test load_compressed_lang_feat=False，评测端 Procrustes 吸收）。
        self.target_rotate_matrix = None
        if target_rotate_matrix_path and os.path.exists(target_rotate_matrix_path):
            self.target_rotate_matrix = np.load(target_rotate_matrix_path)["matrix"].astype(
                np.float32
            )
            print(
                f"current: loaded target rotate matrix {self.target_rotate_matrix.shape} "
                f"from {target_rotate_matrix_path}"
            )

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if self.is_train:
                if asset[:-4] not in self.VALID_ASSETS:
                    continue
            else:
                if (
                    asset[:-4] not in self.VALID_ASSETS
                    and asset[:-4] not in self.EVAL_PC_ASSETS
                ):
                    continue
            try:
                data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
            except Exception as e:
                msg = (
                    f"\n🛑  Failed np.load() in ScanNetPPGSDataset\n"
                    f"    file   : {os.path.join(data_path, asset)}\n"
                    f"    scene  : {data_path}\n"
                    f"    reason : {e}\n"
                )
                print(msg, flush=True)
                raise RuntimeError(msg) from e
        data_dict["name"] = name
        # Add scene path for density-invariant training to load SVD files
        data_dict["scene_path"] = data_path

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "pc_coord" in data_dict.keys():
            data_dict["pc_coord"] = data_dict["pc_coord"].astype(np.float32)

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = data_dict["pc_segment"][:, 0].astype(np.int32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)
            # print("color", data_dict["color"].shape)

        if "opacity" in data_dict.keys():
            data_dict["opacity"] = data_dict["opacity"].astype(np.float32)
            data_dict["opacity"] = data_dict["opacity"].reshape(-1, 1)

        if "quat" in data_dict.keys():
            data_dict["quat"] = data_dict["quat"].astype(np.float32)

        if "sh" in data_dict.keys():
            data_dict["sh"] = data_dict["sh"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "scale" in data_dict.keys():
            data_dict["scale"] = (
                data_dict["scale"].astype(np.float32).clip(0, 1.5)
            )  # clip scale max to 1.5

        # Load SVD-compressed language features if enabled (BEFORE checking lang_feat.npy)
        # This fixes the chicken-and-egg bug where SVD loading required lang_feat.npy to exist
        if self.load_compressed_lang_feat:
            svd_file = os.path.join(data_path, f"lang_feat_grid_svd_r{self.svd_rank}.npz")
            if os.path.exists(svd_file):
                try:
                    svd_data = np.load(svd_file)
                    compressed = svd_data['compressed']  # [M, rank]
                    compressed = canonicalize_svd_sign(compressed)  # 每列最大绝对值取正，消除逐场景基符号歧义
                    # 2026-08-03 全局符号对齐（加载时应用，不修改原始文件）：
                    # max-abs 规范只对强维稳定，弱维符号跨 chunk 随机 → 多 chunk
                    # 训练梯度抵消。用参考 chunk 的类均值符号约定统一所有 chunk。
                    if self.global_signs is not None and name in self.global_signs:
                        compressed = compressed * self.global_signs[name]
                    # 2026-08-04 目标正交旋转（rebuttal Table 2 ③）：等价基扰动，
                    # 必须放在 canonicalize/global_signs 之后（否则会被规范步骤抵消）。
                    if self.target_rotate_matrix is not None:
                        compressed = compressed @ self.target_rotate_matrix
                    if self.svd_center:
                        compressed = remove_scene_mean(compressed)
                    indices = svd_data['indices']  # [N] - point to grid mapping

                    # Add point_to_grid mapping to data_dict (for density-invariant training)
                    data_dict["point_to_grid"] = indices.astype(np.int64)

                    # Expand grid-level features to point-level: [N, rank]
                    point_lang_feat = compressed[indices]

                    # Set lang_feat from SVD (for valid points)
                    # FilterValidPoints will skip arrays with different lengths,
                    # so compressed lang_feat will be preserved (not filtered)
                    # After filtering, both coord and lang_feat will have num_valid points
                    data_dict["lang_feat"] = point_lang_feat.astype(np.float32)
                    # print(f"current: {name} loaded SVD-{self.svd_rank} compressed lang_feat: {point_lang_feat.shape}")
                except Exception as e:
                    print(f"Warning: Failed to load SVD file for {name}: {e}")
            else:
                print(f"Warning: SVD file not found for {name}: {svd_file}")

        # If lang_feat.npy exists and we didn't load SVD features, use the original
        if "lang_feat" in data_dict.keys() and "point_to_grid" not in data_dict.keys():
            data_dict["lang_feat"] = data_dict["lang_feat"].astype(np.float32)

        if "valid_feat_mask" in data_dict.keys():
            # Use the real valid_feat_mask as-is (distinguishes valid from zero features)
            data_dict["valid_feat_mask"] = data_dict["valid_feat_mask"].astype(bool)
            num_points = data_dict["valid_feat_mask"].shape[0]
            num_valid = data_dict["valid_feat_mask"].sum()
            # print(f"current: {name} valid_feat_mask: {num_valid}/{num_points} ({num_valid/num_points*100:.1f}%)")
        else:
            print("current:", name)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][:, 0].astype(np.int32)
            elif "segment200" in data_dict.keys():  # temp update
                data_dict["segment"] = (
                    data_dict.pop("segment200").reshape([-1]).astype(np.int32)
                )
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                try:
                    data_dict["instance"] = data_dict["instance"][:, 0].astype(np.int32)
                except:
                    data_dict["instance"] = (
                        data_dict.pop("instance").reshape([-1]).astype(np.int32)
                    )
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError

        if self.sample_tail:
            # use data_dict["sampled_index"] to denote tail classes are sampled
            tail_mask = np.isin(data_dict["segment"], self.TAIL_CLASSES)
            data_dict["sampled_index"] = np.where(tail_mask)[0]
        return data_dict
