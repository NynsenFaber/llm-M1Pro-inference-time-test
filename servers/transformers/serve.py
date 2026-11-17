"""`transformers serve`, but falls back to the tokenizer when the processor can't load.

shortcut: transformers 5.23 `serve` always loads `AutoProcessor`, which for the text-only
google/gemma-3-1b-it resolves to the multimodal Gemma3Processor and fails (no
preprocessor_config.json). For text models a tokenizer is what AutoProcessor normally
returns. Drop this file once `transformers serve` handles it upstream.
"""

from transformers import AutoTokenizer
from transformers.cli.serving.model_manager import ModelManager
from transformers.cli.transformers import main

_load_processor = ModelManager._load_processor


def _load_processor_or_tokenizer(self, model_id_and_revision, gguf_file=None):
    try:
        return _load_processor(self, model_id_and_revision, gguf_file=gguf_file)
    except OSError:
        model_id, revision = model_id_and_revision.split("@", 1)
        return AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=self.trust_remote_code
        )


ModelManager._load_processor = _load_processor_or_tokenizer
main()  # same CLI as `transformers`: run as `python serve.py serve <model> ...`
