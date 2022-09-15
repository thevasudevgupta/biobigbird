from typing import List, Literal

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxBigBirdForMaskedLM

from biobigbird.constants import HF_TOKEN

PIPELINES_ALIAS = {}


def register_pipeline(name):
    def _register(cls):
        cls.name = name
        PIPELINES_ALIAS[name] = cls
        return cls

    return _register


@register_pipeline("fill-mask")
class FlaxFillMaskPipeline:
    def __init__(self, model, tokenizer, jit_model=False):
        if jit_model:
            model = jax.jit(model)

        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, string: List[str], max_length=512) -> List[str]:
        # TODO: this might not be very efficient
        inputs = self.tokenizer(
            string,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="jax",
        )
        print(jnp.sum(inputs["attention_mask"], axis=-1))

        logits = self.model(**inputs).logits

        mask_positions = inputs.input_ids == self.tokenizer.mask_token_id
        num_masks = jnp.sum(mask_positions, axis=-1)
        assert jnp.alltrue(num_masks == 1), num_masks

        outputs = jnp.argmax(logits[mask_positions], axis=-1)
        outputs = self.tokenizer.batch_decode(outputs)

        return outputs


def pipeline(
    name: str,
    model: str,
    attention_type: Literal["block_sparse", "original_full"] = "block_sparse",
    jit_model: bool = False,
    use_auth_token: str = HF_TOKEN,
):
    assert name in PIPELINES_ALIAS
    model_id = model

    model = FlaxBigBirdForMaskedLM.from_pretrained(
        model_id, attention_type=attention_type, use_auth_token=use_auth_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=use_auth_token)

    pipeline_cls = PIPELINES_ALIAS[name]
    return pipeline_cls(model=model, tokenizer=tokenizer, jit_model=jit_model)


if __name__ == "__main__":
    model_id = "vasudevgupta/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract"
    pipe = pipeline("fill-mask", model=model_id)

    # string = '[MASK] is the tyrosine kinase inhibitor.'
    string = """In mammals, the chromoprotein makes up about 96% of the red blood cells' dry content (by weight), and around 35% of the total content (including water).[5] Hemoglobin has an oxygen-binding capacity of 1.34 mL O2 per gram,[6] which increases the total blood oxygen capacity seventy-fold compared to dissolved oxygen in blood. The mammalian hemoglobin molecule can bind (carry) up to four oxygen molecules.[7]

Hemoglobin is involved in the transport of other gases: It carries some of the body's respiratory carbon dioxide (about 20–25% of the total)[8] as carbaminohemoglobin, in which CO2 is bound to the heme protein. The molecule also carries the important regulatory molecule nitric oxide bound to a thiol group in the globin protein, releasing it at the same time as oxygen.[9]

Hemoglobin is also found outside red blood cells and their progenitor lines. Other cells that contain hemoglobin include the A9 dopaminergic neurons in the substantia nigra, macrophages, alveolar cells, lungs, retinal pigment epithelium, hepatocytes, mesangial cells in the kidney, endometrial cells, cervical cells and vaginal epithelial cells.[10] In these tissues, hemoglobin has a non-oxygen-carrying function as an antioxidant and a regulator of iron metabolism.[11] Excessive glucose in one's blood can attach to hemoglobin and raise the level of hemoglobin A1c.[12]

Hemoglobin and hemoglobin-like molecules are also found in many invertebrates, fungi, and plants.[13] In these organisms, hemoglobins may carry oxygen, or they may act to transport and regulate other small molecules and ions such as carbon dioxide, nitric oxide, hydrogen sulfide and sulfide. A variant of the molecule, called leghemoglobin, is used to scavenge oxygen away from anaerobic systems, such as the nitrogen-fixing nodules of leguminous plants, lest the oxygen poison (deactivate) the system.

Hemoglobinemia is a medical condition in which there is an excess of hemoglobin in the blood plasma. This is an effect of [MASK] hemolysis, in which hemoglobin separates from red blood cells, a form of anemia.

There is more than one hemoglobin gene: in humans, hemoglobin A (the main form of hemoglobin present in adults) is coded for by the genes, HBA1, HBA2, and HBB.[28] The hemoglobin subunit alpha 1 and alpha 2 are coded by the genes HBA1 and HBA2, respectively, which are both on chromosome 16 and are close to each other. The hemoglobin subunit beta is coded by HBB gene which is on chromosome 11 . The amino acid sequences of the globin proteins in hemoglobins usually differ between species. These differences grow with evolutionary distance between species. For example, the most common hemoglobin sequences in humans, bonobos and chimpanzees are completely identical, without even a single amino acid difference in either the alpha or the beta globin protein chains.[29][30][31] Whereas the human and gorilla hemoglobin differ in one amino acid in both alpha and beta chains, these differences grow larger between less closely related species.


"""
    string = " ".join(string.split())
    print(len(string) // 4)

    # intravascular
    print(pipe(string, max_length=768))
