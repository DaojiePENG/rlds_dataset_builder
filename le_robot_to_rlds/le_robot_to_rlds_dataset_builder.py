from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image  # 用于图像尺寸调整


class LeRobotToRLDS(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for converting LeRobot .npy datasets to RLDS format (fixed image shape)."""

    VERSION = tfds.core.Version('1.0.5')
    RELEASE_NOTES = {
        '1.0.5': 'Fixed image shape mismatch (channel order and resolution).',
        '1.0.4': 'Fixed timestamp dtype mismatch.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        # 修正图像形状定义（与实际数据匹配，此处以错误提示的(180, 320, 3)为例，可根据实际调整）
        IMAGE_SHAPE = (224, 224, 3)  # 改为实际图像的高×宽×通道（通道最后）
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'language_instruction': tfds.features.Text(),
                    'observation': tfds.features.FeaturesDict({
                        'wrist_image': tfds.features.Image(
                            shape=IMAGE_SHAPE,
                            dtype=np.uint8,
                            encoding_format='png'
                        ),
                        'image': tfds.features.Image(
                            shape=IMAGE_SHAPE,
                            dtype=np.uint8,
                            encoding_format='png'
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32
                    ),
                    'reward': tfds.features.Scalar(dtype=np.float32),
                    'discount': tfds.features.Scalar(dtype=np.float32),
                    'is_first': tfds.features.Scalar(dtype=np.bool_),
                    'is_last': tfds.features.Scalar(dtype=np.bool_),
                    'is_terminal': tfds.features.Scalar(dtype=np.bool_),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(),
                    'timestamp': tfds.features.Scalar(dtype=np.float32),
                    'frame_index': tfds.features.Scalar(dtype=np.int64),
                })
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        return {
            'train': self._generate_examples(path='data/lerobot/droid_1.0.1/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        def _parse_example(episode_path):
            episode_data = np.load(episode_path, allow_pickle=True).tolist()
            if not episode_data:
                return None

            language_instruction = episode_data[0].get('language_instruction', 'no_instruction')
            first_frame = episode_data[0]
            episode_timestamp = first_frame.get('timestamp', 0.0)
            frame_index = first_frame.get('frame_index', 0.0)

            rlds_steps = []
            for step_idx, step_data in enumerate(episode_data):
                # 图像处理：转换通道顺序+确保形状匹配
                def process_image(img):
                    # 1. 处理通道顺序：若为通道优先(3, h, w)，转为通道最后(h, w, 3)
                    if img.ndim == 3 and img.shape[0] == 3:  # 假设第一维是通道
                        img = np.transpose(img, (1, 2, 0))
                    
                    # 2. 确保数据类型为uint8
                    if img.dtype != np.uint8:
                        if img.dtype == np.float32:
                            img = np.clip(img, 0.0, 1.0)
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # 3. 调整尺寸以匹配定义（如需保持原始尺寸，可跳过此步但需同步修改_info中的形状）
                    target_shape = self._info().features['steps']['observation']['image'].shape
                    if img.shape[:2] != target_shape[:2]:  # 只比较高和宽
                        img = np.array(Image.fromarray(img).resize(
                            (target_shape[1], target_shape[0]),  # resize的参数是(width, height)
                            Image.Resampling.LANCZOS
                        ))
                    
                    return img

                # 提取并处理图像
                wrist_image = step_data.get('observation.images.wrist_left', np.zeros((3, 180, 320), dtype=np.float32))
                main_image = step_data.get('observation.images.exterior_1_left', np.zeros((3, 180, 320), dtype=np.float32))

                observation = {
                    'wrist_image': process_image(wrist_image),
                    'image': process_image(main_image),
                    'state': step_data.get('observation.state', np.zeros(8, dtype=np.float32))
                }

                rlds_step = {
                    'language_instruction': language_instruction,
                    'observation': observation,
                    'action': step_data.get('action', np.zeros(8, dtype=np.float32)),
                    'reward': step_data.get('reward', 0.0),
                    'discount': step_data.get('discount', 1.0),
                    'is_first': step_data.get('is_first', step_idx == 0),
                    'is_last': step_data.get('is_last', step_idx == len(episode_data)-1),
                    'is_terminal': step_data.get('is_terminal', False),
                }
                rlds_steps.append(rlds_step)

            return episode_path, {
                'steps': rlds_steps,
                'episode_metadata': {
                    'file_path': episode_path,
                    'timestamp': episode_timestamp,
                    'frame_index': frame_index,
                }
            }

        episode_paths = glob.glob(path)
        for path in episode_paths:
            yield _parse_example(path)