"""PDF ingestion and text chunking."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

import config


@dataclass
class TextChunk:
    """A chunk of text from a document."""
    content: str
    document: str
    page: int
    chunk_id: str


@dataclass
class ExtractedImage:
    """An image extracted from a document."""
    path: Path
    document: str
    page: int
    image_id: str


def extract_from_pdf(pdf_path: Path) -> tuple[list[str], list[Path]]:
    """Extract text blocks and images from a PDF."""
    doc = fitz.open(pdf_path)
    doc_name = pdf_path.stem
    
    text_blocks = []
    image_paths = []
    seen_hashes = set()
    
    for page_num, page in enumerate(doc):
        # Extract text
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                if len(text) >= config.MIN_CHUNK_LENGTH:
                    text_blocks.append({
                        "content": text,
                        "document": doc_name,
                        "page": page_num + 1
                    })
        
        # Extract images
        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                
                img_bytes = pix.tobytes("png")
                img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
                
                # Skip duplicates
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                
                # Save image
                img_path = config.IMAGES_DIR / f"{doc_name}_p{page_num + 1}_{img_hash}.png"
                pix.save(str(img_path))
                
                image_paths.append({
                    "path": img_path,
                    "document": doc_name,
                    "page": page_num + 1
                })
                
            except Exception:
                continue
    
    doc.close()
    return text_blocks, image_paths


def chunk_text(text: str, size: int = None, overlap: int = None) -> list[str]:
    """Split text into overlapping chunks."""
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if len(text) <= size:
        return [text] if len(text) >= config.MIN_CHUNK_LENGTH else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + size
        
        # Try to break at sentence boundary
        if end < len(text):
            for sep in ['. ', '! ', '? ', '\n', ', ', ' ']:
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if len(chunk) >= config.MIN_CHUNK_LENGTH:
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def process_pdfs(pdf_dir: Path = None) -> tuple[list[TextChunk], list[ExtractedImage]]:
    """Process all PDFs in a directory."""
    pdf_dir = pdf_dir or config.PDF_DIR
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {pdf_dir}")
        return [], []
    
    all_chunks = []
    all_images = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        text_blocks, images = extract_from_pdf(pdf_path)
        
        # Chunk text blocks
        for i, block in enumerate(text_blocks):
            chunks = chunk_text(block["content"])
            for j, text in enumerate(chunks):
                all_chunks.append(TextChunk(
                    content=text,
                    document=block["document"],
                    page=block["page"],
                    chunk_id=f"{block['document']}_p{block['page']}_c{i}_{j}"
                ))
        
        # Add images
        for img in images:
            all_images.append(ExtractedImage(
                path=img["path"],
                document=img["document"],
                page=img["page"],
                image_id=img["path"].stem
            ))
        
        print(f"  → {len(text_blocks)} text blocks, {len(images)} images")
    
    print(f"Total: {len(all_chunks)} chunks, {len(all_images)} images")
    return all_chunks, all_images
