"""CPU-only checks for the README launch examples (run.sh 已移除, 改为两个 docker run 示例)."""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 两个 docker run 示例都应包含的 27B base 关键参数(与旧 run.sh 27B 配置一致)
REQUIRED = (
    'vllm-turing Qwen/Qwen3.8-27B-FP8',
    '--gpus all',
    '--shm-size 16g',
    '--api-key "$VLLM_API_KEY"',
    '--tensor-parallel-size 4',
    '--gdn-prefill-backend flashqla_sm75',
    '--kv-cache-dtype fp8_e4m3',
    '-e VLLM_FIREFLY=1',
    '-e VLLM_FIREFLY_AR=auto',
    '--auto-sleep-idle-timeout 30 --auto-sleep-offload-target exit',
    '--kv-transfer-config ',
    'root/.cache/modelscope',
    'root/.cache/huggingface',
    'root/.cache/vllm',
    'root/.cache/flashinfer',
    'root/.triton/cache',
    'root/.cache/torch_extensions',
)


class LaunchExamples(unittest.TestCase):
    def _blocks(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        return [b for b in text.split('```') if 'docker run -d' in b]

    def test_two_examples(self):
        blocks = self._blocks()
        self.assertEqual(len(blocks), 2, 'README 应有挂载 / 不挂载两个 docker run 示例')

    def test_unversioned_image(self):
        for block in self._blocks():
            self.assertIn('vllm-turing ', block)
            self.assertNotIn('vllm-turing:', block, '镜像不做正式发布, 不应带 :tag')

    def test_required_flags(self):
        for block in self._blocks():
            for needle in REQUIRED:
                self.assertIn(needle, block, f'missing: {needle}')

    def test_mount_example_differs(self):
        blocks = self._blocks()
        mounted = [b for b in blocks if 'vllm-Turing:/vllm-Turing' in b]
        unmounted = [b for b in blocks if 'vllm-Turing:/vllm-Turing' not in b]
        self.assertEqual(len(mounted), 1, '恰好一个示例挂载本地项目')
        self.assertEqual(len(unmounted), 1, '恰好一个示例不挂载本地项目')


if __name__ == '__main__':
    unittest.main()
