import logging

import transformers

from transformers.modeling_longformer import LongformerForSequenceClassification

# class MyLongformer(transformers.modeling_bert.BertPreTrainedModel):
#     config_class = LongformerConfig
#     base_model_prefix = "longformer"


longformer = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096",
                                                                cache_dir="./cache/",
                                                                )

print( longformer )
