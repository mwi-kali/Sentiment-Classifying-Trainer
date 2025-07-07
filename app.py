import fitz
import requests
import torch


import gradio as gr


from bs4 import BeautifulSoup
from sentiment_classifying_trainer.config import Settings
from sentiment_classifying_trainer.preprocess import TextPreprocessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)


def extract_text_from_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" \n ")


def main():
    settings = Settings()
    general_dir = "./trained_models/general_models/checkpoint-1000"
    financial_dir = "./trained_models/financial_models/checkpoint-384"

    print(f"Loading General model from {general_dir} …")
    general_tokenizer = AutoTokenizer.from_pretrained(general_dir)
    general_model = AutoModelForSequenceClassification.from_pretrained(general_dir)
    general_model.eval()

    print(f"Loading Financial model from {financial_dir} …")
    financial_tokenizer = AutoTokenizer.from_pretrained(financial_dir)
    financial_model = AutoModelForSequenceClassification.from_pretrained(financial_dir)
    financial_model.eval()

    def predict_fn(
        model_choice: str,
        text: str,
        upload,
        url: str,
        clean: bool
    ) -> dict:
        content = ""
        if text and not upload and not url:
            content = text
        elif upload and not text and not url:
            path = upload.name
            if path.lower().endswith(".pdf"):
                content = extract_text_from_pdf(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
        elif url and not text and not upload:
            try:
                content = extract_text_from_url(url)
            except Exception as e:
                return {"error": f"Failed to fetch URL: {e}"}
        else:
            return {"error": "Use exactly one input method: Text, File, or URL."}

        if clean:
            pre = TextPreprocessor(settings)
            content = pre.clean_text(content)

        if model_choice == "General":
            tokenizer = general_tokenizer
            model = general_model
        else:
            tokenizer = financial_tokenizer
            model = financial_model

        inputs = tokenizer(
            content,
            padding="max_length",
            truncation=True,
            max_length=settings.max_length,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        idx = int(probs.argmax().item())
        label = settings.id2label[idx]
        confidence = float(probs[idx].item())
        return {label: confidence}

    blocks = gr.Blocks()
    with blocks:
        gr.Markdown("# Sentiment Classifier")
        model_selector = gr.Radio(
            choices=["General", "Financial"],
            value="General",
            label="Choose classifier"
        )

        entry_method = gr.Radio(
            choices=["Text", "File", "URL"],
            value="Text",
            label="Select input method"
        )
        text_input = gr.Textbox(lines=3, placeholder="Enter text here...", label="Text Input")
        file_input = gr.File(label="Upload file (PDF or TXT)", file_count="single", type="file", visible=False)
        url_input = gr.Textbox(lines=1, placeholder="https://example.com", label="URL Input", visible=False)

        clean = gr.Checkbox(label="Clean text?")
        predict_btn = gr.Button("Analyze")
        output = gr.Label(num_top_classes=settings.num_labels)

        def update_visibility(choice):
            return {
                text_input: gr.update(visible=(choice == "Text")),
                file_input: gr.update(visible=(choice == "File")),
                url_input: gr.update(visible=(choice == "URL"))
            }

        entry_method.change(fn=update_visibility, inputs=entry_method, outputs=[text_input, file_input, url_input])

        predict_btn.click(
            fn=predict_fn,
            inputs=[model_selector, text_input, file_input, url_input, clean],
            outputs=output
        )

    blocks.launch()


if __name__ == "__main__":
    main()
