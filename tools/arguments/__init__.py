#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
import yaml


class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    """
    Custom YAML loader that ignores unknown tags instead of raising an error.
    This allows loading YAML files with custom Python object tags.
    """
    def construct_python_object(self, tag_suffix, node):
        # For unknown tags, return None instead of raising an error
        if isinstance(node, yaml.MappingNode):
            return self.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return self.construct_sequence(node)
        else:
            return self.construct_scalar(node)

    def construct_undefined(self, node):
        # Handle undefined tags by returning None
        if isinstance(node, yaml.MappingNode):
            return self.construct_mapping(node)
        elif isinstance(node, yaml.SequenceNode):
            return self.construct_sequence(node)
        else:
            return self.construct_scalar(node)


# Register the custom constructor for undefined tags
SafeLoaderIgnoreUnknown.add_constructor(
    'tag:yaml.org,2002:python/object/apply',
    SafeLoaderIgnoreUnknown.construct_python_object
)
SafeLoaderIgnoreUnknown.add_constructor(
    'tag:yaml.org,2002:python/object/new',
    SafeLoaderIgnoreUnknown.construct_python_object
)
SafeLoaderIgnoreUnknown.add_constructor(
    'tag:yaml.org,2002:python/object',
    SafeLoaderIgnoreUnknown.construct_python_object
)
SafeLoaderIgnoreUnknown.add_constructor(
    None,
    SafeLoaderIgnoreUnknown.construct_undefined
)

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False

        self._language_features_name = "language_features"
        self._feature_level = 2
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        # Handle source_path - may not exist for gsplat models
        if hasattr(g, 'source_path') and g.source_path:
            g.source_path = os.path.abspath(g.source_path)
        try :
            g.lf_path = os.path.join(g.source_path, g.language_features_name)
        except:
            pass
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError, OSError):
        print("Config file not found, using command line arguments only")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def get_combined_args_from_yaml(parser: ArgumentParser, param_groups: list = None):
    """
    Load arguments from cfg.yml (gsplat format) and merge with command line arguments.
    Ensures all default values are included for missing parameters.

    Args:
        parser: ArgumentParser instance
        param_groups: List of ParamGroup instances to get default values from

    Returns:
        Namespace with merged arguments from config file and command line
    """
    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    # Get default values from param_groups if provided
    defaults = {}
    if param_groups:
        for param_group in param_groups:
            for key, value in vars(param_group).items():
                if not key.startswith("_"):
                    defaults[key] = value
                else:
                    # For attributes starting with _, store without underscore
                    defaults[key[1:]] = value

    # Try to load cfg.yml from model_path
    cfg_dict = {}
    if hasattr(args_cmdline, 'model_path') and args_cmdline.model_path:
        cfg_yml_path = os.path.join(args_cmdline.model_path, "cfg.yml")
        cfg_args_path = os.path.join(args_cmdline.model_path, "cfg_args")

        # Try cfg.yml first (gsplat format)
        if os.path.exists(cfg_yml_path):
            print(f"Loading config from: {cfg_yml_path}")
            try:
                with open(cfg_yml_path, 'r') as f:
                    # Use custom loader that ignores unknown tags
                    cfg_data = yaml.load(f, Loader=SafeLoaderIgnoreUnknown)

                # Map gsplat config keys to our parameter names
                # gsplat uses different naming, so we need to map them
                if cfg_data:
                    # Common gsplat configuration mappings
                    key_mapping = {
                        # gsplat -> our naming
                        'data_dir': 'source_path',
                        'model_path': 'model_path',
                        'output_dir': 'model_path',
                        'white_background': 'white_background',
                        'sh_degree': 'sh_degree',
                        'images': 'images',
                        'resolution': 'resolution',
                        'eval': 'eval',
                        'data_device': 'data_device',
                    }

                    def extract_simple_values(data, cfg_dict, key_mapping):
                        """Recursively extract simple (serializable) values from nested data."""
                        if isinstance(data, dict):
                            for key, value in data.items():
                                if value is None or isinstance(value, (str, int, float, bool, list)):
                                    mapped_key = key_mapping.get(key, key)
                                    cfg_dict[mapped_key] = value
                                elif isinstance(value, dict):
                                    # Recursively process nested dicts
                                    extract_simple_values(value, cfg_dict, key_mapping)
                                # Skip complex objects (strategies, optimizers, etc.)

                    extract_simple_values(cfg_data, cfg_dict, key_mapping)
                    print(f"Loaded {len(cfg_dict)} parameters from cfg.yml")
            except Exception as e:
                print(f"Warning: Could not parse cfg.yml: {e}")
                print("Using command line arguments and defaults only")
                cfg_dict = {}

        # Try cfg_args (original Gaussian Splatting format) as fallback
        elif os.path.exists(cfg_args_path):
            print(f"Loading config from: {cfg_args_path}")
            with open(cfg_args_path, 'r') as f:
                cfgfile_string = f.read()
            args_cfgfile = eval(cfgfile_string)
            cfg_dict = vars(args_cfgfile)
        else:
            print("No config file found, using command line arguments and defaults")

    # Merge: defaults -> config file -> command line
    merged_dict = defaults.copy()  # Start with defaults
    merged_dict.update(cfg_dict)  # Override with config file values

    # Override with command line arguments (if not None)
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)
