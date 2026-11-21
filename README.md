# Serve Model · public beta

Turn a PEFT sequence-classification adapter into a private-key inference endpoint. The endpoint runs in your own Google Cloud project. You can also export the same container for another host.

**Agent quickstart:** [Deployment contract and machine-readable commands](llms.txt). Deploy PEFT LoRA BERT-tiny sequence classifiers to a bearer-authenticated HTTPS prediction API, or export a Docker image build context with ONNX inference.

## Install

Use Python 3.10–3.12 in a virtual environment. This beta bundles a pinned Google BERT base and a small demonstration adapter. Installation downloads PyTorch and the ONNX export tools.

    python3.10 -m venv .venv
    source .venv/bin/activate
    python -m pip install "git+https://github.com/A5omics/serve-model-beta.git@v0.1.0b3"

## Deploy

Install the Google Cloud CLI, sign in with gcloud auth login, and choose a billing-enabled project you can deploy from source into. Enable Cloud Run, Cloud Build and Artifact Registry and configure the source-build service account with Cloud Run Builder permissions. Use a new service name on first deployment; redeploying that name updates its production revision. Your project pays its own usage and artifact storage. Minimum instances is zero and maximum instances is two.

    serve-model deploy sample --project YOUR_PROJECT --service my-support-router

Replace sample with /path/to/your-adapter to deploy your model. The command exports the model, builds the container, deploys it and compares a live authenticated prediction with local inference. Anonymous requests must return 401. It prints the verified URL and a local key-file path. Preserve the key file for redeployments.

    curl -sS "$MODEL_URL" -H "Authorization: Bearer $MODEL_API_KEY" \
      -H 'Content-Type: application/json' -d '{"text":"My dashboard keeps crashing"}'

Set MODEL_URL to the printed HTTPS URL including /predict. Set MODEL_API_KEY to the file contents in your calling application's server-side secret store. The URL is reachable over the internet. The bearer key gates inference.

## Custom domain and other hosts

Route a hostname you own to the service and wait for HTTPS to be active. Then add --domain models.example.com to test and print the custom URL. Domain registration, DNS and certificate issuance are separate from the CLI. Google documents its supported routes in the [Cloud Run custom-domain guide](https://cloud.google.com/run/docs/mapping-custom-domains).

For any Docker host:

    serve-model bundle sample --out ./model-container
    docker build -t my-support-router ./model-container
    docker run --rm -p 127.0.0.1:8080:8080 my-support-router

POST to http://127.0.0.1:8080/predict with the bearer key. The build context contains the exported model and a key digest. Keep model images and registries private if your weights are proprietary.

## Adapter contract

Also supported: prajjwal1/bert-tiny at revision 6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837. Its base weights download on first use. Adapters can supply ordered class names via config.json id2label.

Bundled base: google/bert_uncased_L-2_H-128_A-2, revision 30b0a37ccaaa32f332884b96992754e246e48c5f. Supply adapter_config.json, adapter_model.safetensors (LoRA SEQ_CLS with classifier head), and labels.json, an ordered list of class names. Examples: ["billing", "technical"]. Inputs truncate to 512 tokens. The sample was trained on 36 synthetic support messages. No accuracy or uptime SLA is claimed.

The [base model](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) and its license are included. To stop a Cloud Run deployment, delete its service in your project and remove unneeded build images separately.

## Try the app integration locally

On Linux or macOS run `serve-model local sample` in a foreground terminal. After export and a reference prediction check it emits one JSON receipt with the loopback URL and key-file path. Call the URL from another process. Ctrl-C stops the service. See the [support classifier model card](src/serve_model/sample/README.md) for the request format.
