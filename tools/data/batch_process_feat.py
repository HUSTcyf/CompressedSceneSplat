import os

device = 2
src_dim = 768
src_dim = 16
iteration = 30000

# dataset = "lerf_ovs"
# scenes = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
# feat_levels = [0]
# res = 1

dataset = "3DOVS"
scenes = ['lawn', 'bench', 'sofa', 'room', 'bed']
feat_levels = [0]
res = 4

# dataset = "replica"
# scenes = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4', 'room_0', 'room_1', 'room_2']
# feat_levels = [0]
# res = 1

src_path = f"./datasets/{dataset}"
dst_path = f"./gaussian_results/{dataset}"
npy_path = f"./gaussian_train/{dataset}"
# dst_path = "/new_data/cyf/projects/OccamLGS/output/LERF"
# feat_levels = [1]
for scene in scenes:
    scene_ = ''.join(scene.split('_'))
    for f in feat_levels:
        # cmd = f"CUDA_VISIBLE_DEVICES={device} python tools/data/gaussian_feature_extractor.py -s {src_path}/{scene} -m {dst_path}/max_0_depth_True_default_{scene_} --iteration {iteration} --eval --feature_level {f} --src_dim {src_dim} --save_npy {npy_path}/max_0_depth_True_default_{scene_}/ -r {res} --use_ply"
        # print(cmd)
        # os.system(cmd)

        cmd = f"CUDA_VISIBLE_DEVICES={device} python tools/data/feature_map_renderer.py -m {dst_path}/{scene} --iteration {iteration} --eval --skip_train --feature_level {f} --src_dim {src_dim} -r {res}" \
        f" --lang_checkpoint /new_data/cyf/projects/SceneSplat/output_features/{scene}/checkpoint_with_features_p.pth --visualize --use_siglip_sam2_format"
        # cmd = f"CUDA_VISIBLE_DEVICES={device} python tools/data/feature_map_renderer.py -m {dst_path}/{scene} --iteration {iteration} --eval --skip_train --feature_level {f} --src_dim {src_dim} -r {res}" \
        # f" --lang_checkpoint /new_data/cyf/projects/SceneSplat/output_features/{scene}/checkpoint_with_features_s.pth --visualize"
        print(cmd)
        os.system(cmd)

        # cmd = f"CUDA_VISIBLE_DEVICES={device} python tools/assign_2d_labels_to_3d.py --dataset_type {dataset} \
        # --model_path /new_data/cyf/projects/SceneSplat/gaussian_results/{dataset}/{scene} \
        # --source_path /new_data/cyf/projects/SceneSplat/datasets/{dataset}/{scene} \
        # --output_dir /new_data/cyf/projects/SceneSplat/gaussian_train/{dataset}/train/{scene} --eval -r {res}"
        # print(cmd)
        # os.system(cmd)
    # break
