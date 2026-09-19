"""CPU-only checks for the unified vllm-turing build context."""
import ast
import re
import unittest
import tempfile
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / 'docker' / 'Dockerfile'


class BuildContract(unittest.TestCase):
    def test_official_prebuilt_base(self):
        text = DOCKERFILE.read_text(encoding='utf-8')
        self.assertIn('vllm/vllm-openai:v0.29.0-cu129@sha256:', text)
        script = (ROOT / 'docker/build.sh').read_text(encoding='utf-8')
        self.assertNotIn('git clone', script)
        self.assertNotIn('git init', script)
        self.assertNotIn('prepare_upstream', script)

    def test_entrypoint_contract(self):
        # 新流程: 容器入口是 /opt/vllm-turing/entrypoint.sh(拉最新+按需重新 overlay),
        # 最终 exec vllm serve。不再继承官方 ENTRYPOINT 直接 serve。
        text = DOCKERFILE.read_text(encoding='utf-8')
        self.assertIn('ENTRYPOINT ["/opt/vllm-turing/entrypoint.sh"]', text)
        entrypoint = (ROOT / 'docker/entrypoint.sh').read_text(encoding='utf-8')
        self.assertIn('exec vllm serve', entrypoint)
        self.assertIn('https://github.com/javier-house/vllm-Turing', entrypoint)
        self.assertIn('https://gitee.com/javier_house/vllm-Turing', entrypoint)
        # 大版本不一致时只提醒、不覆盖
        self.assertIn('UPSTREAM_VERSION', entrypoint)
        installer = (ROOT / 'docker/install_sm75_overlay.py').read_text(encoding='utf-8')
        for path in ('entrypoints/cli/main.py', 'entrypoints/cli/serve.py', 'entrypoints/serve/entry.py'):
            self.assertNotIn('"' + path + '"', installer)

    def test_installer(self):
        install = runpy.run_path(str(ROOT / 'docker/install_speculative.py'))['install']
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp) / 'site-packages/vllm'
            evidence = Path(tmp) / 'evidence/speculative-files.json'
            result = install(ROOT / 'docker/speculative', package, evidence)
            self.assertEqual(len(result), 8)
            self.assertTrue(evidence.is_file())
            self.assertTrue((package.parent / 'sm75_fa2_graph.py').is_file())

    def test_copied_inputs_exist(self):
        text = DOCKERFILE.read_text(encoding='utf-8')
        text = text.replace('\\\n', ' ')
        for line in text.splitlines():
            if line.startswith('COPY ') and '--from=' not in line:
                for source in line.split()[1:-1]:
                    self.assertTrue((ROOT / source).exists(), source)

    def test_unified_final_target(self):
        text = DOCKERFILE.read_text(encoding='utf-8')
        self.assertEqual(re.findall(r'^FROM .* AS (.*)$', text, re.M)[-1], 'final')
        self.assertIn('python3 /opt/vllm-turing/install_speculative.py', text)
        # 构建入口产出未版本化的 vllm-turing 镜像(不做正式发布), 指向单一 Dockerfile
        build = (ROOT / 'docker/build.sh').read_text(encoding='utf-8')
        self.assertIn('docker/Dockerfile', build)
        self.assertIn('--target final', build)
        self.assertRegex(build, r'--tag\s+vllm-turing(\s|$)')
        self.assertNotIn('v0.1.4', build)

    def test_source_syntax_and_activation(self):
        for path in (ROOT / 'docker/speculative').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        text = (ROOT / 'docker/speculative/vllm/v1/worker/gpu/model_runner.py').read_text(encoding='utf-8')
        for hook in ('_sm75_fa2_graph.install()', '_sm75_gdn_meta.install()'):
            self.assertEqual(text.count(hook), 1)

    def test_no_host_cache_copy(self):
        text = DOCKERFILE.read_text(encoding='utf-8')
        for value in ('FROM local/', 'COPY flashinfer-cache', '/mnt/user/'):
            self.assertNotIn(value, text)
        self.assertIsNone(re.search(r'(?<!\d)10(?:\.\d{1,3}){3}(?!\d)', text))


if __name__ == '__main__':
    unittest.main()
