import os
import numpy as np

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class Matterport3DGSDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "valid_feat_mask",
        "normal",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment"]

    def __init__(
        self,
        multilabel=False,
        is_train=True,
        load_compressed_lang_feat=False,
        svd_rank=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel
        self.is_train = is_train
        self.load_compressed_lang_feat = load_compressed_lang_feat
        self.svd_rank = svd_rank

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
                    f"\n🛑  Failed np.load() in Matterport3DGSDataset\n"
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

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)
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
                    print(f"current: {name} loaded SVD-{self.svd_rank} compressed lang_feat: {point_lang_feat.shape}")
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
            print(f"current: {name} valid_feat_mask: {num_valid}/{num_points} ({num_valid/num_points*100:.1f}%)")
        else:
            print("current:", name)

        if "segment" in data_dict.keys():
            # 21 classes processed by pointcept repo
            data_dict["segment"] = (
                data_dict.pop("segment").reshape([-1]).astype(np.int32)
            )
        elif "segment_nyu_160" in data_dict.keys():
            # top presented 160 nyu classes
            data_dict["segment"] = (
                data_dict.pop("segment_nyu_160").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "pc_segment" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment").reshape([-1]).astype(np.int32)
            )
        elif "pc_segment_nyu_160" in data_dict.keys():
            data_dict["pc_segment"] = (
                data_dict.pop("pc_segment_nyu_160").reshape([-1]).astype(np.int32)
            )

        return data_dict


@DATASETS.register_module()
class Matterport3D_160_GSDataset(Matterport3DGSDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "segment_nyu_160",
        "instance",
        "quat",
        "scale",
        "opacity",
        "lang_feat",
        "valid_feat_mask",
        "normal",
    ]
    EVAL_PC_ASSETS = ["pc_coord", "pc_segment_nyu_160"]
