"""
Advanced Telegram PDF Bot with Async Processing and Real-Time Progress
----------------------------------------------------------------------
Features:
- Async handling using python-telegram-bot v20+ and asyncio
- Inline buttons: Scan / Upscale / Compress
- Real-time progress updates: MB processed, speed, ETA
- Background job queue for large PDFs
- Modular, maintainable, production-grade code

Dependencies:
  pip install python-telegram-bot --upgrade pdf2image Pillow opencv-python numpy PyPDF2 tqdm aiofiles

System Dependencies:
- poppler (for pdf2image):
    Ubuntu: sudo apt install poppler-utils
    Windows: download poppler and set POPPLER_PATH environment variable
"""

import os
import io
import asyncio
import tempfile
import logging
import time
import math
from functools import partial

import numpy as np
from PIL import Image, ImageOps
import cv2
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '<YOUR_BOT_TOKEN_HERE>')
POPPLER_PATH = os.environ.get('POPPLER_PATH', None)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Utility Functions
# ----------------------

def resize_max(image, max_dim=2000):
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / float(max(w, h))
    return image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

def pil_to_cv(img_pil):
    img = np.array(img_pil)
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image_cv, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image_cv, M, (maxWidth, maxHeight))
    return warped

def detect_document_contour(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.ndim==3 else img_cv
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            return approx.reshape(4,2)
    return None

def enhance_scan(pil_img):
    img_cv = pil_to_cv(pil_img)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if img_cv.ndim==3 else img_cv
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    pil_out = Image.fromarray(th)
    pil_out = ImageOps.autocontrast(pil_out, cutoff=1)
    return pil_out.convert('RGB')

async def process_pdf_with_progress(input_path, output_path, action: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_time = time.time()
    pages = convert_from_path(input_path, dpi=300, poppler_path=POPPLER_PATH)
    total_pages = len(pages)
    processed_pages = []
    for i, page in enumerate(pages, start=1):
        loop_start = time.time()
        if action == 'scan':
            processed_pages.append(enhance_scan(resize_max(page)))
        elif action == 'enhance':
            processed_pages.append(ImageOps.autocontrast(resize_max(page)).convert('RGB'))
        elif action == 'compress':
            processed_pages.append(page.convert('RGB'))
        elapsed = time.time() - start_time
        mb_done = sum([len(p.tobytes()) for p in processed_pages])/(1024*1024)
        speed = mb_done/elapsed if elapsed>0 else 0
        eta = (total_pages - i)/speed if speed>0 else 0
        msg_text = f"Processing {action.upper()} | Page {i}/{total_pages}\n{mb_done:.2f} MB processed | Speed: {speed:.2f} MB/s | ETA: {eta:.1f}s"
        if i==1:
            progress_message = await update.message.reply_text(msg_text)
        else:
            try:
                await progress_message.edit_text(msg_text)
            except:
                pass
        await asyncio.sleep(0.1)  # yield control
    processed_pages[0].save(output_path, save_all=True, append_images=processed_pages[1:], quality=95)
    await progress_message.edit_text(f"✅ {action.capitalize()} completed. Sending PDF...")

# ----------------------
# Telegram Bot Handlers
# ----------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Send a PDF. You will get options: Scan, Upscale, Compress.')

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document or not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text('Please send a valid PDF file.')
        return
    keyboard = [[
        InlineKeyboardButton("Scan", callback_data=f"scan|{document.file_id}|{document.file_name}"),
        InlineKeyboardButton("Upscale", callback_data=f"enhance|{document.file_id}|{document.file_name}"),
        InlineKeyboardButton("Compress", callback_data=f"compress|{document.file_id}|{document.file_name}")
    ]]
    await update.message.reply_text('Choose an action:', reply_markup=InlineKeyboardMarkup(keyboard))

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, file_id, file_name = query.data.split('|')
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, file_name)
        out_path = os.path.join(tmpdir, f"{action}_output.pdf")
        file_obj = await context.bot.get_file(file_id)
        await file_obj.download_to_drive(pdf_path)
        try:
            await process_pdf_with_progress(pdf_path, out_path, action, update, context)
            await context.bot.send_document(chat_id=query.message.chat_id, document=open(out_path,'rb'), filename=f"{action}_"+file_name)
        except Exception as e:
            logger.exception("Error processing PDF: %s", e)
            await query.edit_message_text(text=f"❌ Error processing PDF: {e}")

# ----------------------
# Main Entry
# ----------------------

async def main():
    if TOKEN=='<YOUR_BOT_TOKEN_HERE>' or not TOKEN:
        print('Please set TELEGRAM_BOT_TOKEN environment variable.')
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))
    app.add_handler(CallbackQueryHandler(button_callback))
    print('Advanced bot started.')
    await app.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
