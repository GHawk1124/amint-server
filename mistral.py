from mistralai import Mistral
import os
import asyncio
from dotenv import load_dotenv

def get_ocr_result(api_key: str, file_path: str) -> str:
    client = Mistral(api_key=api_key)
    uploaded_file = client.files.upload(
        file={
            "file_name": file_path,
            "content": open(file_path, "rb"),
        },
        purpose="ocr"
    )
    retrieved_file = client.files.retrieve(file_id=uploaded_file.id)
    signed_url = client.files.get_signed_url(file_id=retrieved_file.id)
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )
    final_text = ""
    for idx, page in enumerate(ocr_response.pages):
        final_text += f"Page {idx+1}:\n{page.markdown}\n\n"
    return final_text


if __name__ == "__main__":
    # get api key from .env file
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    file_path = "./lbm.pdf"
    ocr_result = get_ocr_result(api_key, file_path)
    with open("ocr_result.txt", "w", encoding="utf-8") as f:
        f.write(ocr_result)
