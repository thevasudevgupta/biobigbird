import jax
import jax.numpy as jnp
from transformers import FlaxAutoModelForMaskedLM, AutoTokenizer, FlaxBigBirdForMaskedLM
from biobigbird.constants import HF_TOKEN

from typing import List

PIPELINES_ALIAS = {}


def register_pipeline(name):    
    def _register(cls):
        cls.name = name
        PIPELINES_ALIAS[name] = cls
        return cls
    return _register


@register_pipeline("fill-mask")
class FlaxFillMaskPipeline:
    def __init__(self, model, tokenizer, _jit_model=False):
        if _jit_model:
            model = jax.jit(model)

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, string: List[str]) -> List[str]:
        # TODO: this might not be very efficient
        inputs = self.tokenizer(string, max_length=512, padding="max_length", truncation=True, return_tensors="jax")
        
        logits = self.model(**inputs).logits

        mask_positions = inputs.input_ids == self.tokenizer.mask_token_id
        num_masks = jnp.sum(mask_positions, axis=-1)
        assert jnp.alltrue(num_masks == 1), num_masks

        outputs = jnp.argmax(logits[mask_positions], axis=-1)
        outputs = self.tokenizer.batch_decode(outputs)

        return outputs


def pipeline(name: str, model: str, attention_type="original_full", use_auth_token=HF_TOKEN):
    assert name in PIPELINES_ALIAS
    model_id = model

    model = FlaxBigBirdForMaskedLM.from_pretrained(model_id, attention_type=attention_type, use_auth_token=use_auth_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=use_auth_token)

    pipeline_cls = PIPELINES_ALIAS[name]
    return pipeline_cls(model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    model_id = 'vasudevgupta/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract'
    pipe = pipeline('fill-mask', model=model_id)

    string = '[MASK] is the tyrosine kinase inhibitor.'
    print(pipe(string))
