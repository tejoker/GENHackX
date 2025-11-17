from docx import Document

# Path to the .docx file (update if needed)
doc_path = "data/WIP - GenHack   Kayrros data User Guide.docx"

def print_docx_content(path):
    doc = Document(path)
    print(f"Reading: {path}\n")
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            print(text)

if __name__ == "__main__":
    print_docx_content(doc_path)
