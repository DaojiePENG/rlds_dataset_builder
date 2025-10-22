import os
import numpy as np
import tqdm
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotToNPYConverter:
    """将LeRobot格式数据集转换为.npy格式的转换器类"""
    
    def __init__(self, repo_id, output_dir=None, episodes=None):
        self.repo_id = repo_id
        self.episodes = episodes
        self.output_dir = output_dir or f"data/{repo_id}"
        self.dataset = None  # 数据集对象
        self.features = None  # 要提取的特征列表

    def load_dataset(self, print_info=True):
        """
        加载LeRobot数据集并打印基本信息
        
        Args:
            print_info: 是否打印数据集信息
        """
        print(f"Loading dataset: {self.repo_id}")
        self.dataset = LeRobotDataset(self.repo_id, episodes=self.episodes)
        
        if print_info:
            print(f"Selected episodes: {self.dataset.episodes}")
            print(f"Number of episodes selected: {self.dataset.num_episodes}")
            print(f"Total frames selected: {self.dataset.num_frames}")
            print(f"All available features in dataset:")
            all_features = list(self.dataset.features.keys())
            for i, feat in enumerate(all_features, 1):
                print(f"  {i}. {feat}")
        
        return self.dataset

    def extract_features(self, frame):
        """
        提取帧数据中的特征（可被继承重写）
        
        子类需重写此方法以自定义特征提取逻辑，需返回包含指定特征的字典
        
        Args:
            frame: 单帧数据（来自LeRobotDataset）
        
        Returns:
            dict: 提取的特征字典
        """
        # 默认提取示例特征（可根据需求修改或在子类中重写）
        selected_features = {
            # 语言指令
            'language_instruction',
            # 观测特征（嵌套结构）
            'observation.images.exterior_1_left',
            'observation.images.wrist_left', 
            'observation.state',
            # 动作特征
            'action',
            # 奖励和折扣
            'reward', 'discount',
            # 核心标记特征
            'is_first', 'is_last', 'is_terminal',
            # 元数据
            'timestamp', 'frame_index'
        }
        
        frame_data = {}
        for feature in selected_features:
            try:
                if isinstance(frame[feature], torch.Tensor):
                    frame_data[feature] = frame[feature].cpu().numpy()
                else:
                    frame_data[feature] = frame[feature]
            except (KeyError, AttributeError) as e:
                print(f"Warning: Could not access feature '{feature}': {e}")
                continue
        return frame_data

    def convert_and_save(self):
        """使用自定义的特征提取方法转换并保存数据集"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        selected_episodes = self.dataset.episodes
        
        # 转换每个episode
        for episode_idx in tqdm.tqdm(selected_episodes, desc="Converting episodes"):
            # 获取episode的起止帧索引
            from_idx = self.dataset.meta.episodes["dataset_from_index"][episode_idx]
            to_idx = self.dataset.meta.episodes["dataset_to_index"][episode_idx]
            episode_length = to_idx - from_idx
            print(f"\nProcessing episode {episode_idx} (frames: {episode_length})")
            
            # 提取该episode的所有帧数据
            episode_data = []
            for frame_idx in tqdm.tqdm(
                range(from_idx, to_idx), 
                desc=f"Episode {episode_idx} frames", 
                leave=False
            ):
                frame = self.dataset[frame_idx]
                # 使用自定义的特征提取方法
                frame_data = self.extract_features(frame)
                episode_data.append(frame_data)
            
            # 保存为npy文件
            save_path = os.path.join(self.output_dir, f"episode_{episode_idx}.npy")
            np.save(save_path, episode_data)
            print(f"Saved episode {episode_idx} to {save_path}")
        
        print(f"\nSuccessfully converted dataset to {self.output_dir}")


# 示例：继承重写特征提取方法
class CustomFeatureConverter(LeRobotToNPYConverter):
    """自定义特征提取器示例"""
    def extract_features(self, frame):
        """只提取机器人状态、动作和语言指令"""
        custom_features = {
            # 语言指令
            'language_instruction',
            # 观测信息
            'observation.images.exterior_1_left',
            'observation.images.wrist_left',
            'observation.state',
            # 动作值
            'action',
            # 奖励和折扣
            'reward',
            'discount',
            # 核心标记特征
            'is_first', 
            'is_last', 
            'is_terminal',
            # 元数据
            'timestamp',
            'frame_index',
        }
        
        frame_data = {}
        for feature in custom_features:
            try:
                val = frame[feature]
                frame_data[feature] = val.cpu().numpy() if isinstance(val, torch.Tensor) else val
            except Exception as e:
                print(f"Warning: Missing feature '{feature}': {e}")
        return frame_data


if __name__ == "__main__":
    # 示例1：使用默认特征提取器
    print("=== Using default feature extractor ===")
    converter = LeRobotToNPYConverter(
        repo_id="lerobot/droid_1.0.1",
        episodes=list(range(0, 3))  # 转换前3个episode
    )
    converter.load_dataset()
    converter.convert_and_save()
    
    # 示例2：使用自定义特征提取器
    print("\n=== Using custom feature extractor ===")
    custom_converter = CustomFeatureConverter(
        repo_id="lerobot/droid_1.0.1",
        output_dir="data/custom_droid_features",  # 自定义输出目录
        episodes=list(range(3, 5))  # 转换第3-4个episode
    )
    custom_converter.load_dataset()
    custom_converter.convert_and_save()