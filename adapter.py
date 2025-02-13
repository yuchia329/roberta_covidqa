from adapters import list_adapters

# source can be "ah" (AdapterHub), "hf" (hf.co) or None (for both, default)
adapter_infos = list_adapters(source="hf", model_name="FacebookAI/roberta-base")