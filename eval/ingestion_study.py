"""Check a known refundable period survives actual Office and image ingestion."""

import json
import sys
import tempfile
from pathlib import Path

from app.config.settings import Settings
from app.core.enums import DocumentCategory
from app.retrieval.unstructured_parser import parse_path


def main(out):
    from docx import Document
    from PIL import Image, ImageDraw, ImageFont

    text = "Quality claim window is 15 days."
    results = {}
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        document = Document()
        document.add_heading("Policy A", level=1)
        document.add_paragraph(text)
        document.save(root / "policy.docx")
        image = Image.new("RGB", (950, 125), "white")
        font = (
            ImageFont.truetype("arial.ttf", 42)
            if sys.platform == "win32"
            else ImageFont.load_default(size=42)
        )
        ImageDraw.Draw(image).text((15, 24), text, fill="black", font=font)
        image.save(root / "policy.png")
        for path in sorted(root.iterdir()):
            chunks = parse_path(
                path,
                document_id="fixture",
                title="Policy A",
                category=DocumentCategory.FAQ,
                settings=Settings(
                    unstructured_strategy="ocr_only", unstructured_ocr_languages="eng"
                ),
            )
            results[path.suffix] = {
                "chunks": len(chunks),
                "contains_fact": "15 days" in " ".join(c.content for c in chunks),
                "parsers": sorted({c.metadata.get("parser", "") for c in chunks}),
            }
    Path(out).write_text(json.dumps(results, indent=2), encoding="utf8")


if __name__ == "__main__":
    main(sys.argv[1])
