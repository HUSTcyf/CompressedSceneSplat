#!/usr/bin/env python3
"""
PyTorch Checkpoint参数量分析工具 (使用torchinfo)

依赖:
    pip install torchinfo

使用方法:
    # 基本分析（仅state_dict，无需config）
    python tools/analyze_checkpoint.py --checkpoint path/to/model.pth

    # 完整分析（需要config和project-root，加载模型实例并使用torchinfo）
    python tools/analyze_checkpoint.py -c model.pth \\
        --config configs/custom/lang-pretrain-litept-ovs-gridsvd.py \\
        --project-root /path/to/project

    # 详细模式
    python tools/analyze_checkpoint.py -c model.pth --verbose

    # 对比两个checkpoint
    python tools/analyze_checkpoint.py -c model1.pth --compare model2.pth

    # 导出JSON报告
    python tools/analyze_checkpoint.py -c model.pth --output report.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torchinfo import summary


class CheckpointAnalyzer:
    """Checkpoint分析器（使用torchinfo）"""

    def __init__(self, ckpt_path: str):
        self.ckpt_path = Path(ckpt_path)
        self.ckpt = None
        self.state_dict = None
        self.metadata = {}
        self._load()

    def _load(self):
        """加载checkpoint"""
        print(f"加载: {self.ckpt_path}")

        self.ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)

        # 提取state_dict
        if isinstance(self.ckpt, dict) and 'state_dict' in self.ckpt:
            self.state_dict = self.ckpt['state_dict']
            self.metadata = {k: v for k, v in self.ckpt.items()
                            if k != 'state_dict' and not isinstance(v, dict)}
        elif isinstance(self.ckpt, dict) and 'model' in self.ckpt:
            self.state_dict = self.ckpt['model']
            self.metadata = {k: v for k, v in self.ckpt.items()
                            if k != 'model' and not isinstance(v, dict)}
        else:
            self.state_dict = self.ckpt

        print(f"  ✓ state_dict keys: {len(self.state_dict)}")

    def analyze(self) -> Dict:
        """分析checkpoint"""
        # 统计参数
        total_params = 0
        module_stats = defaultdict(lambda: {'params': 0, 'buffers': 0})

        for key, value in self.state_dict.items():
            if isinstance(value, torch.Tensor):
                num = value.numel()
                total_params += num

                # 按模块分组
                module = key.rsplit('.', 1)[0] if '.' in key else 'root'
                if 'running_mean' in key or 'running_var' in key or 'num_batches' in key:
                    module_stats[module]['buffers'] += num
                else:
                    module_stats[module]['params'] += num

        # 推断模型信息
        model_info = self._infer_model_info()

        return {
            'path': str(self.ckpt_path),
            'name': self.ckpt_path.name,
            'total_params': total_params,
            'params_m': round(total_params / 1e6, 2),
            'module_stats': dict(module_stats),
            'metadata': self.metadata,
            'model_info': model_info,
            'num_keys': len(self.state_dict),
        }

    def _infer_model_info(self) -> Dict:
        """推断模型结构信息"""
        info = {
            'input_channels': 'N/A',
            'output_channels': 'N/A',
            'architecture': 'Unknown',
            'encoder_stages': [],
            'decoder_stages': [],
        }

        # 检测输入通道（stem层）
        for key, value in self.state_dict.items():
            if 'stem' in key and 'weight' in key:
                shape = value.shape
                if len(shape) >= 2:
                    info['input_channels'] = int(shape[1])
                break

        # 检测输出通道（最后的投影层）
        for key in reversed(list(self.state_dict.keys())):
            if ('proj' in key or 'fc' in key) and 'weight' in key:
                shape = self.state_dict[key].shape
                if len(shape) >= 2:
                    info['output_channels'] = int(shape[0])
                break

        # 检测编码器/解码器阶段
        for key in self.state_dict.keys():
            if 'backbone.enc.enc' in key or '.enc.enc' in key:
                parts = key.split('.')
                for p in parts:
                    if p.startswith('enc') and p[3:].isdigit():
                        stage = p[3:]
                        if stage not in info['encoder_stages']:
                            info['encoder_stages'].append(stage)
                        break
            if 'backbone.dec.dec' in key or '.dec.dec' in key:
                parts = key.split('.')
                for p in parts:
                    if p.startswith('dec') and p[3:].isdigit():
                        stage = p[3:]
                        if stage not in info['decoder_stages']:
                            info['decoder_stages'].append(stage)
                        break

        # 排序阶段
        info['encoder_stages'] = sorted(info['encoder_stages'], key=lambda x: int(x) if x.isdigit() else 0)
        info['decoder_stages'] = sorted(info['decoder_stages'], key=lambda x: int(x) if x.isdigit() else 0)

        # 推断架构类型
        keys_str = ' '.join(self.state_dict.keys())
        if 'attn' in keys_str or 'transformer' in keys_str.lower():
            info['architecture'] = 'Transformer'
        elif 'conv' in keys_str.lower():
            info['architecture'] = 'CNN'
        elif 'point' in keys_str.lower():
            info['architecture'] = 'Point-based'

        return info

    def print_report(self, data: Dict, verbose: bool = False, top_n: int = 15):
        """打印分析报告"""
        print(f"\n{'='*80}")
        print(f"Checkpoint 分析: {data['name']}")
        print(f"{'='*80}\n")

        # 基本信息
        print("参数统计:")
        print("-" * 80)
        p = data['total_params']
        print(f"  总参数量:     {p:,} ({p/1e6:.2f}M)")
        print(f"  State keys:   {data['num_keys']}")
        print()

        # 模型信息
        info = data['model_info']
        print("模型结构:")
        print("-" * 80)
        print(f"  架构类型:     {info['architecture']}")
        print(f"  输入通道:     {info['input_channels']}")
        print(f"  输出维度:     {info['output_channels']}")
        if info['encoder_stages']:
            print(f"  编码器阶段:   {info['encoder_stages']}")
        if info['decoder_stages']:
            print(f"  解码器阶段:   {info['decoder_stages']}")
        print()

        # 元数据
        if data['metadata']:
            print("元数据:")
            print("-" * 80)
            for key, value in sorted(data['metadata'].items()):
                if key == 'optimizer':
                    print(f"  {key}: <optimizer state>")
                elif not isinstance(value, (dict, list)):
                    print(f"  {key}: {value}")
            print()

        # 模块详情
        print(f"主要模块 (Top {top_n}):")
        print("-" * 80)
        print(f"{'模块':<50} {'参数量':>15} {'占比':>10}")
        print("-" * 80)

        sorted_modules = sorted(
            data['module_stats'].items(),
            key=lambda x: -x[1]['params']
        )

        for module, stats in sorted_modules[:top_n]:
            params = stats['params']
            pct = params / p * 100
            print(f"{module:<50} {params:>12,} ({params/1e6:5.2f}M) {pct:>9.1f}%")

        print()

        # 内存估算
        print("内存占用估算 (FP32):")
        print("-" * 80)
        param_mb = p * 4 / (1024**2)
        grad_mb = param_mb
        opt_mb = param_mb * 2  # AdamW
        print(f"  模型参数:      {param_mb:.1f} MB")
        print(f"  梯度:          {grad_mb:.1f} MB")
        print(f"  优化器:        {opt_mb:.1f} MB")
        print(f"  训练总显存:    {param_mb + grad_mb + opt_mb:.1f} MB")
        print()


class CheckpointAnalyzerWithModel:
    """Checkpoint分析器（加载模型实例，使用torchinfo.summary）"""

    def __init__(self, ckpt_path: str, config_path: str, project_root: str):
        self.ckpt_path = Path(ckpt_path)
        self.config_path = Path(config_path)
        self.project_root = Path(project_root)
        self.model = None
        self.ckpt = None
        self._load()

    def _load(self):
        """加载checkpoint和模型"""
        print(f"加载配置: {self.config_path}")
        print(f"项目根目录: {self.project_root}")

        # Add project to path (must be before importing pointcept)
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        # Import build_model first
        from pointcept.models.builder import build_model

        # Load config
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", self.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Build model from config
        self.model = build_model(config_module.model)

        print(f"  ✓ 模型类型: {type(self.model).__name__}")

        # Load checkpoint weights
        print(f"加载checkpoint: {self.ckpt_path}")
        self.ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)

        if isinstance(self.ckpt, dict) and 'state_dict' in self.ckpt:
            state_dict = self.ckpt['state_dict']
        elif isinstance(self.ckpt, dict) and 'model' in self.ckpt:
            state_dict = self.ckpt['model']
        else:
            state_dict = self.ckpt

        # Remove 'module.' prefix if present (DDP checkpoints)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        print(f"  ✓ 权重已加载")

    def print_summary(self):
        """
        使用torchinfo打印模型结构摘要（仅显示参数，不执行forward）
        """
        print(f"\n{'='*100}")
        print(f"torchinfo 模型结构摘要")
        print(f"{'='*100}\n")

        # 使用torchinfo的summary功能，仅显示模型结构，不执行实际推理
        summary(
            self.model,
            col_names=["num_params", "params_percent"],
            depth=3,
            row_settings=["var_names"],
            col_width=20,
        )


def compare(data1: Dict, data2: Dict):
    """对比两个checkpoint"""
    print(f"\n{'='*100}")
    print(f"Checkpoint 对比")
    print(f"{'='*100}\n")

    name1 = data1['name'][:35]
    name2 = data2['name'][:35]

    print(f"{'指标':<20} {name1:<40} {name2:<40}")
    print("-" * 100)

    p1, p2 = data1['total_params'], data2['total_params']
    print(f"{'参数量':<20} {f'{p1:,} ({p1/1e6:.1f}M)':<40} {f'{p2:,} ({p2/1e6:.1f}M)':<40}")

    ratio = p1 / p2 if p2 > 0 else float('inf')
    direction = '>' if ratio > 1 else '<'
    print(f"{'比例':<20} {f'{ratio:.2f}x ({direction})':<40} {'':<40}")

    i1, i2 = data1['model_info'], data2['model_info']
    print(f"{'输入通道':<20} {str(i1['input_channels']):<40} {str(i2['input_channels']):<40}")
    print(f"{'输出维度':<20} {str(i1['output_channels']):<40} {str(i2['output_channels']):<40}")
    print(f"{'编码器阶段':<20} {str(i1['encoder_stages']):<40} {str(i2['encoder_stages']):<40}")
    print(f"{'解码器阶段':<20} {str(i1['decoder_stages']):<40} {str(i2['decoder_stages']):<40}")
    print()


def export_json(data: Dict, output_path: str):
    """导出JSON报告"""
    # 准备导出数据（移除不可序列化的内容）
    export_data = {
        'path': data['path'],
        'name': data['name'],
        'total_params': int(data['total_params']),
        'params_million': data['params_m'],
        'model_info': data['model_info'],
        'num_keys': data['num_keys'],
        'top_modules': [
            {
                'name': module,
                'params': int(stats['params']),
                'params_m': round(stats['params'] / 1e6, 2),
                'percentage': round(stats['params'] / data['total_params'] * 100, 2),
            }
            for module, stats in sorted(
                data['module_stats'].items(),
                key=lambda x: -x[1]['params']
            )[:20]
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"✓ 报告已导出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Checkpoint分析 (torchinfo)')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='checkpoint文件路径')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径 (用于加载完整模型实例)')
    parser.add_argument('--compare', type=str, default=None,
                        help='要对比的另一个checkpoint')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细模式')
    parser.add_argument('--top', type=int, default=15,
                        help='显示前N个模块 (默认: 15)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='导出JSON报告')
    parser.add_argument('--project-root', type=str, default=None,
                        help='项目根目录 (与--config一起使用)')

    args = parser.parse_args()

    # 如果提供了config，使用CheckpointAnalyzerWithModel进行完整分析
    if args.config:
        if args.project_root is None:
            print("错误: 使用--config时必须同时指定--project-root")
            return 1

        print("\n使用完整模型模式 (需要config)")
        print("-" * 80)

        analyzer_model = CheckpointAnalyzerWithModel(args.checkpoint, args.config, args.project_root)
        analyzer_model.print_summary()

    # 始终执行基础分析 (state_dict分析)
    print("\n使用State Dict分析")
    print("-" * 80)
    analyzer = CheckpointAnalyzer(args.checkpoint)
    data = analyzer.analyze()
    analyzer.print_report(data, verbose=args.verbose, top_n=args.top)

    # 导出报告
    if args.output:
        export_json(data, args.output)

    # 对比模式
    if args.compare:
        other = CheckpointAnalyzer(args.compare)
        other_data = other.analyze()
        compare(data, other_data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
