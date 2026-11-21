# Support message classifier

This LoRA adapter for google/bert_uncased_L-2_H-128_A-2 routes English customer-support messages to billing or technical support. It was trained on 36 synthetic examples. Scores are softmax outputs. Accuracy on real customer traffic has not been measured.

## Run inference from an app

The adapter includes a trained classification head and ordered labels. Run the following in an activated Python 3.10–3.12 environment on macOS or Linux, from the folder containing this card:

    pip install "git+https://github.com/A5omics/serve-model-beta.git@v0.1.0b3"
    serve-model local . --service support-trial

Keep the process running. Its JSON readiness receipt contains endpoint and key_file. The endpoint is loopback HTTP. In a separate app process, read the key file locally and POST {"text":"Please refund a duplicate charge"} to the receipt's endpoint with headers Content-Type: application/json and Authorization: Bearer YOUR_KEY. Missing credentials return HTTP 401.

For remote HTTPS on your own authorized Google Cloud project:

    serve-model deploy . --project YOUR_PROJECT --service support-trial --json

Cloud hosting is billed to that project. The same request contract applies. See the [deployment guide](https://github.com/A5omics/serve-model-beta).
