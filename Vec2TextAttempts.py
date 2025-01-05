import vec2text
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

inversion_model = vec2text.models.InversionModel.from_pretrained("jxm/gtr__nq__32")
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained("jxm/gtr__nq__32__correct")

corrector = vec2text.load_corrector(inversion_model, corrector_model)

def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer) -> torch.Tensor:

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=128,
                       truncation=True,
                       padding="max_length",).to("cuda")

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings


encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
corrector = vec2text.load_pretrained_corrector("gtr-base")

embeddings = get_gtr_embeddings([
       "Jack Morris is a PhD student at Cornell Tech in New York City",
       "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity"
], encoder, tokenizer)

vec2text.invert_embeddings(
    embeddings=embeddings.cuda(),
    corrector=corrector,
    num_steps=20,
)
