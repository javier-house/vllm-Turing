"""Install the audited speculative overlay without host cache dependencies."""
import compileall
import hashlib
import json
import shutil
from pathlib import Path

def install(source, package, evidence):
    records = {}
    for src in sorted(source.rglob('*.py')):
        rel = src.relative_to(source)
        dest = package / Path(*rel.parts[1:]) if rel.parts[0] == 'vllm' else package.parent / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        records[rel.as_posix()] = hashlib.sha256(dest.read_bytes()).hexdigest()
        if not compileall.compile_file(str(dest), quiet=1):
            raise RuntimeError(f'Syntax check failed: {rel}')
    if len(records) != 8:
        raise RuntimeError('Expected eight reviewed speculative source files')
    runner_path = package / 'v1/worker/gpu/model_runner.py'
    runner = runner_path.read_text(encoding='utf-8')
    for hook in ('_sm75_fa2_graph.install()', '_sm75_gdn_meta.install()'):
        if hook not in runner:
            raise RuntimeError(f'Missing activation hook: {hook}')

    # PP+MTP: install_sm75_overlay 先对 model_runner.py 注入 sample_tokens 的
    # encoder_cache 守卫(无 encoder 的 PP rank 跳过 MM gather), 但本脚本随后整文件
    # 覆盖为带 SM75 fa2/gdn hook 的 spec 版, 会抹掉该注入。此处补回(幂等, 锚点唯一):
    # 否则 PP>1 时 drafter 所在 last rank(encoder_cache=None) 仍走 gather_mm_embeddings
    # -> AttributeError: ... no attribute 'encoder_runner'。
    _anchor = "        if self.speculator is not None and self.speculator.supports_mm_inputs:\n"
    _probe = "                and self.encoder_cache is not None):\n"
    if _probe not in runner:
        if runner.count(_anchor) != 1:
            raise RuntimeError(
                f"PP+MTP guard anchor not unique in model_runner.py "
                f"(count={runner.count(_anchor)})"
            )
        runner = runner.replace(
            _anchor,
            "        if (self.speculator is not None and self.speculator.supports_mm_inputs\n"
            "                and self.encoder_cache is not None):\n",
        )
        runner_path.write_text(runner, encoding='utf-8')
        if not compileall.compile_file(str(runner_path), quiet=1):
            raise RuntimeError('Syntax check failed: model_runner.py (post-PP+MTP-guard)')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


if __name__ == '__main__':
    import vllm
    install(Path('/opt/vllm-turing/speculative'), Path(vllm.__file__).resolve().parent,
            Path('/opt/vllm-turing/evidence/speculative-files.json'))
