import base64
import re
from pathlib import Path

try:
    import fitz

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

from .models import Assignment, AttachmentContext
from .output import sanitize_filename


MAX_PDF_VISION_PAGES = 12


def attachment_is_solution_reference(filename: str) -> bool:
    stem = Path(filename or "").stem.casefold()
    return bool(re.search(r"(?:^|[_\-\s])(sol|solution|solutions|פתרון|פתרונות)(?:$|[_\-\s])", stem))


def attachment_is_pdf(filename: str, content_type: str) -> bool:
    return filename.lower().endswith(".pdf") or "pdf" in content_type.lower()


def render_pdf_to_png_bytes(pdf_bytes: bytes, max_pages: int = MAX_PDF_VISION_PAGES) -> list[bytes]:
    if not pdf_bytes or not PYMUPDF_AVAILABLE:
        return []

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: list[bytes] = []

    for page_index in range(min(len(document), max_pages)):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        images.append(pixmap.tobytes("png"))

    document.close()
    return images


def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not pdf_bytes or not PYMUPDF_AVAILABLE:
        return ""

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages = [document.load_page(index).get_text("text").strip() for index in range(len(document))]
    finally:
        document.close()
    return "\n\n".join(page for page in pages if page).strip()


def render_pdf_to_image_data_urls(pdf_bytes: bytes, max_pages: int = MAX_PDF_VISION_PAGES) -> list[str]:
    image_urls: list[str] = []
    for png_bytes in render_pdf_to_png_bytes(pdf_bytes, max_pages=max_pages):
        image_urls.append(f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}")
    return image_urls


def crop_image_to_data_url(
    image_path: Path,
    bounding_box: tuple[float, float, float, float],
) -> str | None:
    if not PYMUPDF_AVAILABLE or not image_path.is_file():
        return None
    document = fitz.open(str(image_path))
    try:
        page = document.load_page(0)
        rect = page.rect
        x0, y0, x1, y1 = bounding_box
        clip = fitz.Rect(rect.x0 + rect.width * x0, rect.y0 + rect.height * y0,
                         rect.x0 + rect.width * x1, rect.y0 + rect.height * y1)
        if clip.is_empty or clip.width < 1 or clip.height < 1:
            return None
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
        png_bytes = pixmap.tobytes("png")
    finally:
        document.close()
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


def collect_attachment_context(client, assignment: Assignment, attachment_dir: Path) -> AttachmentContext:
    readable_extensions = {
        ".txt", ".md", ".csv", ".json", ".py", ".html", ".htm",
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".java", ".sh", ".ps1",
        ".asm", ".s", ".sql", ".js", ".ts", ".go", ".rs", ".yaml", ".yml",
    }
    readable_filenames = {"makefile", "dockerfile"}
    collected: list[str] = []
    image_urls: list[str] = []
    saved_files: list[dict[str, object]] = []
    preview_image_paths: list[str] = []
    attachment_dir.mkdir(parents=True, exist_ok=True)

    for attachment in assignment.attachments:
        filename = attachment.get("filename", "")
        suffix = Path(filename).suffix.lower()
        solution_reference = attachment_is_solution_reference(filename)
        try:
            file_bytes, content_type = client.download_attachment_bytes(attachment)
        except Exception as exc:
            collected.append(f"Could not read attachment {filename}: {exc}")
            continue

        if suffix in readable_extensions or filename.lower() in readable_filenames:
            safe_stem = sanitize_filename(Path(filename).stem or filename)
            safe_name = safe_stem + suffix
            local_path = attachment_dir / safe_name
            local_path.write_bytes(file_bytes)
            decoded = file_bytes.decode("utf-8", errors="replace")
            source_label = "Reference solution file" if solution_reference else "Source file"
            collected.append(f"{source_label} {filename}:\n{decoded}")
            saved_files.append(
                {
                    "filename": filename,
                    "local_path": str(local_path),
                    "mimetype": content_type or "text/plain",
                    "source_role": "solution_reference" if solution_reference else "assignment",
                    "image_count": 0,
                    "preview_image_paths": [],
                }
            )
            continue

        if attachment_is_pdf(filename, content_type):
            safe_stem = sanitize_filename(Path(filename).stem)
            safe_name = safe_stem + ".pdf"
            local_path = attachment_dir / safe_name
            local_path.write_bytes(file_bytes)
            extracted_text = extract_pdf_text(file_bytes)
            if extracted_text:
                source_label = "REFERENCE SOLUTION PDF" if solution_reference else "Extracted text from PDF"
                collected.append(f"{source_label} {filename}:\n{extracted_text}")
            else:
                collected.append(
                    f"PDF attachment available: {filename}. Use attached page images to read its contents."
                )

            pdf_images = [] if solution_reference else render_pdf_to_png_bytes(file_bytes)
            for index, png_bytes in enumerate(pdf_images, start=1):
                preview_path = attachment_dir / f"{safe_stem}_page_{index}.png"
                preview_path.write_bytes(png_bytes)
                preview_image_paths.append(str(preview_path))
                image_urls.append(f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}")

            saved_files.append(
                {
                    "filename": filename,
                    "local_path": str(local_path),
                    "mimetype": content_type or "application/pdf",
                    "source_role": "solution_reference" if solution_reference else "assignment",
                    "image_count": len(pdf_images),
                    "extracted_text_char_count": len(extracted_text),
                    "preview_image_paths": preview_image_paths[-len(pdf_images) :] if pdf_images else [],
                }
            )

    return AttachmentContext(
        texts=collected,
        image_urls=image_urls,
        saved_files=saved_files,
        preview_image_paths=preview_image_paths,
    )
