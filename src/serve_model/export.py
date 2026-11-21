from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
MODEL="google/bert_uncased_L-2_H-128_A-2"
REVISION="30b0a37ccaaa32f332884b96992754e246e48c5f"

def export(adapter,assets):
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import onnxruntime as ort
    import numpy as np
    config = json.loads((adapter/"adapter_config.json").read_text())
    
    base_id = config.get("base_model_name_or_path")
    revisions = {MODEL: REVISION, "prajjwal1/bert-tiny": "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"}
    if (base_id not in revisions or config.get("revision") not in (None,revisions[base_id])
        or config.get("task_type") != "SEQ_CLS" or config.get("peft_type") != "LORA"
        or "classifier" not in (config.get("modules_to_save") or []) or not (adapter/"adapter_model.safetensors").is_file()):
        raise SystemExit("Expected a compatible BERT classification LoRA with a trained head.")
    if any(adapter.glob("*.bin")):
        raise SystemExit("Supply only safetensors weights")
    if (adapter/"labels.json").exists():
        labels = json.loads((adapter/"labels.json").read_text())
    else:
        mapping = json.loads((adapter/"config.json").read_text())["id2label"]
        labels = [mapping[str(i)] for i in range(len(mapping))]
    if not isinstance(labels,list) or len(labels)<2 or not all(isinstance(s,str) for s in labels) or len(set(labels))!=len(labels):
        raise SystemExit("Invalid class labels")
    base_path = ROOT/"base"
    if base_id != MODEL:
        from huggingface_hub import snapshot_download
        base_path = snapshot_download(base_id, revision=revisions[base_id],
                                      allow_patterns=["config.json","pytorch_model.bin","vocab.txt"])
    tokenizer_path = adapter if (adapter/"tokenizer.json").exists() else base_path
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    base=AutoModelForSequenceClassification.from_pretrained(base_path,num_labels=len(labels),attn_implementation="eager",local_files_only=True)
    merged=PeftModel.from_pretrained(base,str(adapter),local_files_only=True).merge_and_unload().eval()
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=merged
        def forward(self,input_ids):
            return self.model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids)).logits.softmax(dim=-1)
    model=Predictor().eval()
    torch.set_num_threads(1)
    sample=tok("Please refund the second charge",return_tensors="pt").input_ids
    torch.onnx.export(model,(sample,),str(assets/"model.onnx"),input_names=["input_ids"],output_names=["scores"],
                      dynamic_axes={"input_ids":{1:"tokens"}},opset_version=17,dynamo=False)
    sess=ort.InferenceSession(str(assets/"model.onnx"),providers=["CPUExecutionProvider"])
    for text in ["Refund please","The dashboard freezes whenever I try to open a new project"]:
        ids=tok(text,return_tensors="pt").input_ids
        np.testing.assert_allclose(sess.run(None,{"input_ids":ids.numpy()})[0],model(ids).detach().numpy(),atol=0.00001)
    tok.backend_tokenizer.save(str(assets/"tokenizer.json"))
    (assets/"labels.json").write_text(json.dumps(labels))
    return dict(zip(labels,map(float,model(sample).detach().numpy()[0])))
