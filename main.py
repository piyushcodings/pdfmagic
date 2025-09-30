import os
import logging
import asyncio
import time
import uuid
import math
from typing import Dict, Any, Optional

# --- Third-Party Libraries (MUST BE INSTALLED) ---
# System dependency: poppler must be installed for pdf2image to work!
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# PDF/Image Processing Libraries
from pdf2image import convert_from_path
from PIL import Image, ImageChops
import cv2
import numpy as np
from pypdf import PdfReader, PdfWriter # Using pypdf, the modern successor to PyPDF2

# --- Configuration ---
BOT_TOKEN = "7966525096:AAHZz-HMAQb2_9EPmOBvlatqhyP16gQ3UFQ"
LOG_LEVEL = logging.INFO
BASE_DIR = "data"
JOB_QUEUE: asyncio.Queue = asyncio.Queue()  # Global job queue
# Status tracking: {file_id: {status, progress_message_id, start_time, total_bytes, ...}}
PROCESSING_STATUS: Dict[str, Dict[str, Any]] = {} 

# --- Setup Logging ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=LOG_LEVEL,
)
logger = logging.getLogger(__name__)

# --- Utility Functions ---

def setup_directories():
    """Ensure the necessary directories exist."""
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'input'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, 'output'), exist_ok=True)

def get_keyboard() -> InlineKeyboardMarkup:
    """Returns the inline keyboard with PDF processing options."""
    keyboard = [
        [
            InlineKeyboardButton("✨ Scan (Correct & Clean)", callback_data="process_scan"),
        ],
        [
            InlineKeyboardButton("🖼️ Upscale/Enhance (DPI/Contrast)", callback_data="process_upscale"),
        ],
        [
            InlineKeyboardButton("⬇️ Compress (Reduce Size)", callback_data="process_compress"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)

def update_processing_progress(file_id: str, current_step: int):
    """Safely update the progress step."""
    if file_id in PROCESSING_STATUS:
        PROCESSING_STATUS[file_id]['current_step'] = current_step

# --- Real-Time Progress Tracking ---

async def progress_tracker(file_id: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Asynchronous function to send real-time progress updates.
    Runs concurrently with the processing job.
    """
    chat_id = PROCESSING_STATUS[file_id]['chat_id']
    message_id = PROCESSING_STATUS[file_id]['progress_message_id']
    total_bytes = PROCESSING_STATUS[file_id]['total_bytes']
    
    total_mb = total_bytes / (1024 * 1024)
    
    while PROCESSING_STATUS[file_id]['status'] in ['QUEUED', 'PROCESSING']:
        start_time = PROCESSING_STATUS[file_id]['start_time']
        elapsed_time = time.time() - start_time
        
        current_step = PROCESSING_STATUS[file_id].get('current_step', 0)
        total_steps = PROCESSING_STATUS[file_id].get('total_steps', 1) 
        
        percentage = min(100, int((current_step / total_steps) * 100)) 

        if elapsed_time > 0 and percentage > 0 and total_steps > 0:
            # We assume processing is roughly proportional to the number of pages processed
            # This is an estimate, as actual MB/s is complex to track across libraries
            processing_speed_steps_per_sec = current_step / elapsed_time
            
            # Simple linear ETA based on pages
            if percentage < 100:
                steps_remaining = total_steps - current_step
                eta_seconds = steps_remaining / processing_speed_steps_per_sec
                eta_display = f"{math.ceil(eta_seconds)}s"
            else:
                eta_display = "Finishing..."
            
            # Use a static value for processed MB to avoid over-complicating tracking
            progress_text = (
                f"⏳ **Status: {PROCESSING_STATUS[file_id]['status']}**\n"
                f"🚀 **Task:** {PROCESSING_STATUS[file_id]['task_name']}\n"
                f"📊 **Progress:** {percentage}%\n"
                f"📄 **Pages Processed:** {current_step}/{total_steps}\n"
                f"💾 **Total Size:** {total_mb:.2f} MB\n"
                f"⚡ **Speed (Pages/s):** {processing_speed_steps_per_sec:.2f}\n"
                f"🕰️ **ETA:** {eta_display}"
            )
        else:
            progress_text = (
                f"⏳ **Status: {PROCESSING_STATUS[file_id]['status']}**\n"
                f"🚀 **Task:** {PROCESSING_STATUS[file_id]['task_name']}\n"
                f"📊 **Progress:** 0%\n"
                f"💾 **Total Size:** {total_mb:.2f} MB\n"
                f"🕰️ **ETA:** Calculating..."
            )
        
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=progress_text,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.warning(f"Could not update progress message for {file_id}: {e}")
            break
            
        await asyncio.sleep(2) # Update interval
        
    logger.info(f"Progress tracker stopped for {file_id}.")


# --- Core Processing Logic (Synchronous/Blocking) ---

def process_scan_and_clean(input_path: str, output_path: str, file_id: str, dpi: int = 300) -> None:
    """Converts PDF pages to images, applies advanced binarization (scanning), and reassembles."""
    
    # 1. Convert PDF to images
    images = convert_from_path(input_path, dpi=dpi, thread_count=1)
    
    total_pages = len(images)
    processed_images = []

    for i, pil_img in enumerate(images):
        # 2. Convert PIL Image to OpenCV NumPy array
        np_img = np.array(pil_img.convert('RGB')) 
        cv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        # 3. Apply Advanced Binarization (Scan)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding for demonstration. 
        # For *real* document scanning, you'd integrate more complex steps like:
        # - Deskewing (rotation correction)
        # - Contour detection for perspective correction (cv2.getPerspectiveTransform)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Convert back to PIL for re-assembly
        # Note: We save the binarized (single-channel) image as RGB for better compatibility 
        # when re-assembling via PIL, though grayscale is more efficient.
        processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
        processed_images.append(processed_pil)
        
        update_processing_progress(file_id, i + 1)

    # 5. Re-assemble into PDF
    if processed_images:
        processed_images[0].save(
            output_path, 
            save_all=True, 
            append_images=processed_images[1:], 
            resolution=dpi 
        )
    else:
        raise ValueError("No pages were processed.")

def process_upscale_and_enhance(input_path: str, output_path: str, file_id: str, new_dpi: int = 600) -> None:
    """Upscales PDF by converting to high DPI images, enhancing, and reassembling."""
    
    # Increase DPI significantly for "upscaling"
    images = convert_from_path(input_path, dpi=new_dpi, thread_count=1)
    
    total_pages = len(images)
    processed_images = []

    for i, pil_img in enumerate(images):
        # Enhance Contrast (Example using PIL's ImageChops)
        # Apply a simple contrast enhancement (e.g., multiply by a factor)
        factor = 1.2
        processed_img = ImageChops.multiply(pil_img, Image.new(pil_img.mode, pil_img.size, int(255 * factor)))

        processed_images.append(processed_img)
        update_processing_progress(file_id, i + 1)

    # Re-assemble into PDF
    if processed_images:
        processed_images[0].save(
            output_path, 
            save_all=True, 
            append_images=processed_images[1:], 
            resolution=new_dpi 
        )
    else:
        raise ValueError("No pages were processed.")

def process_compress(input_path: str, output_path: str, file_id: str, compression_dpi: int = 150, quality: int = 60) -> None:
    """Compresses PDF by reducing image resolution, lowering JPEG quality, and reassembling."""
    
    # 1. Convert to low-DPI images
    images = convert_from_path(input_path, dpi=compression_dpi, thread_count=1)
    
    total_pages = len(images)
    processed_images = []

    for i, pil_img in enumerate(images):
        # No image manipulation needed other than the low DPI conversion
        processed_images.append(pil_img.convert('RGB')) 
        update_processing_progress(file_id, i + 1)

    # 2. Re-assemble into PDF with low JPEG quality (aggressive compression)
    if processed_images:
        processed_images[0].save(
            output_path, 
            save_all=True, 
            append_images=processed_images[1:], 
            resolution=compression_dpi,
            quality=quality, # PIL's quality parameter for JPEG output
            optimize=True
        )
    else:
        raise ValueError("No pages were processed.")

# --- Asynchronous Job Wrapper ---

async def process_pdf_job(
    input_path: str, output_path: str, task_name: str, file_id: str, context: ContextTypes.DEFAULT_TYPE
) -> Optional[str]:
    """
    Runs the heavy-lifting logic in a separate thread pool to avoid blocking 
    the event loop, then returns the output path.
    """
    logger.info(f"Starting {task_name} job for {file_id}...")
    
    try:
        # Determine the page count for progress tracking
        reader = PdfReader(input_path)
        total_pages = len(reader.pages)
    except Exception as e:
        logger.warning(f"Could not read PDF structure for {file_id}: {e}")
        total_pages = 10 # Default fallback
        
    PROCESSING_STATUS[file_id]['total_steps'] = total_pages
    
    def blocking_processing():
        """This function executes the CPU-bound image manipulation logic."""
        if task_name == "Scan & Clean":
            process_scan_and_clean(input_path, output_path, file_id)
        elif task_name == "Upscale & Enhance":
            process_upscale_and_enhance(input_path, output_path, file_id)
        elif task_name == "Compress":
            process_compress(input_path, output_path, file_id)
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        return output_path

    # Run the blocking function in a separate thread
    loop = asyncio.get_running_loop()
    final_output_path = await loop.run_in_executor(None, blocking_processing) 
    
    return final_output_path

# --- Job Queue Management (Same as previous, omitted for brevity but included in full code) ---
# NOTE: The job_consumer function remains the same, as it handles the async orchestration.

# --- Job Queue Management (Restored for completeness) ---

async def job_consumer(application: Application):
    """
    The background worker that pulls jobs from the queue and executes them.
    This runs indefinitely as an asyncio task.
    """
    logger.info("Job consumer started.")
    while True:
        job = await JOB_QUEUE.get()
        file_id = job['file_id']
        chat_id = job['chat_id']
        task_name = job['task_name']
        input_path = job['input_path']
        output_path = job['output_path']
        context = job['context']
        
        PROCESSING_STATUS[file_id]['status'] = 'PROCESSING'
        PROCESSING_STATUS[file_id]['start_time'] = time.time()

        # Start the progress tracker concurrently
        tracker_task = asyncio.create_task(progress_tracker(file_id, context))
        
        final_output_path = None
        try:
            final_output_path = await process_pdf_job(
                input_path, output_path, task_name, file_id, context
            )
            
            PROCESSING_STATUS[file_id]['status'] = 'SENDING'

            if final_output_path and os.path.exists(final_output_path):
                # Send the final file
                await context.bot.send_document(
                    chat_id=chat_id,
                    document=final_output_path,
                    caption=f"✅ Your **{task_name}** PDF is ready!",
                    filename=os.path.basename(output_path),
                    read_timeout=60, 
                    write_timeout=60,
                    connect_timeout=60,
                    api_kwargs={'parse_mode': 'Markdown'}
                )
                
                # Update final message
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=PROCESSING_STATUS[file_id]['progress_message_id'],
                    text=f"✅ **Job Complete!**\nFile sent successfully.",
                    parse_mode='Markdown'
                )

            else:
                raise RuntimeError("Processing failed to produce a valid output file.")

        except Exception as e:
            logger.error(f"Job processing failed for {file_id} in consumer: {e}")
            PROCESSING_STATUS[file_id]['status'] = 'FAILED'
            
            error_message = f"❌ **Error during {task_name}**:\n"
            if "No pages were processed" in str(e) or "empty document" in str(e):
                 error_message += "The PDF appears corrupt or empty."
            elif "Poppler" in str(e) or "pdf2image" in str(e):
                 error_message += "System dependency (Poppler) is missing or broken on the server."
            else:
                 error_message += "An unexpected error occurred. Try again later."
                 
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=PROCESSING_STATUS[file_id]['progress_message_id'],
                    text=error_message,
                    parse_mode='Markdown'
                )
            except Exception:
                await context.bot.send_message(
                    chat_id=chat_id, 
                    text=error_message, 
                    parse_mode='Markdown'
                )

        finally:
            tracker_task.cancel() # Stop the progress tracker
            
            # Cleanup - auto-delete processed files
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if final_output_path and os.path.exists(final_output_path):
                    os.remove(final_output_path)
            except Exception as e:
                logger.warning(f"Could not delete files for {file_id}: {e}")
            
            del PROCESSING_STATUS[file_id] # Remove tracking status
            JOB_QUEUE.task_done() # Signal that the job is done
            logger.info(f"Job {file_id} finished. Queue size: {JOB_QUEUE.qsize()}")


# --- Telegram Handlers (Same as previous, omitted for brevity but included in full code) ---

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and instructions."""
    await update.message.reply_text(
        "👋 Welcome to PDF Magic Bot!\n\n"
        "Please send me a **PDF file** to get started. I can scan, upscale, or compress it."
    )

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming document messages."""
    document = update.message.document
    
    if not document.mime_type or 'application/pdf' not in document.mime_type:
        await update.message.reply_text(
            "🚫 I can only process **PDF files** for now. Please send a valid PDF."
        )
        return

    if document.file_size > 50 * 1024 * 1024: # 50 MB limit
        await update.message.reply_text(
            "⚠️ File too large. The current processing limit is 50MB."
        )
        return

    file_id = document.file_id
    context.user_data['current_file'] = {
        'file_id': file_id,
        'file_name': document.file_name,
        'file_size': document.file_size
    }

    await update.message.reply_text(
        f"📄 PDF received: **{document.file_name}** ({document.file_size / (1024*1024):.2f} MB).\n\n"
        "Please select a processing option:",
        reply_markup=get_keyboard(),
        parse_mode='Markdown'
    )

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles button presses for processing options."""
    query = update.callback_query
    await query.answer() 
    
    task_action = query.data
    file_info = context.user_data.get('current_file')
    
    if not file_info:
        await query.edit_message_text(
            "🚫 Error: Please send the PDF file again before selecting an option."
        )
        return

    file_id = file_info['file_id']
    original_name = file_info['file_name']
    original_size = file_info['file_size']
    
    # Check if a job for this file is already running (simple check)
    if file_id in PROCESSING_STATUS:
        await query.edit_message_text(
            f"🚫 A job for this file is already in status: {PROCESSING_STATUS[file_id]['status']}"
        )
        return
    
    # Determine the job name and output suffix
    if task_action == "process_scan":
        task_name, suffix = "Scan & Clean", "_scanned.pdf"
    elif task_action == "process_upscale":
        task_name, suffix = "Upscale & Enhance", "_enhanced.pdf"
    elif task_action == "process_compress":
        task_name, suffix = "Compress", "_compressed.pdf"
    else:
        await query.edit_message_text("🚫 Invalid option selected.")
        return

    # 1. Initiate download and setup paths
    downloaded_file = await context.bot.get_file(file_id)
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(BASE_DIR, 'input', f"{unique_id}_original.pdf")
    base_name = os.path.splitext(original_name)[0].replace(' ', '_')
    output_path = os.path.join(BASE_DIR, 'output', f"{base_name}{suffix}")
    
    try:
        await downloaded_file.download_to_drive(input_path)
    except Exception as e:
        logger.error(f"File download failed for {file_id}: {e}")
        await query.edit_message_text(f"❌ Could not download the file. Error: {e}")
        return

    # 2. Inform the user and send initial progress message
    await query.edit_message_text(
        f"✅ Job for **{task_name}** accepted.\n"
        f"⏳ Status: QUEUED\n"
        "You will receive real-time updates shortly...",
        parse_mode='Markdown'
    )
    progress_message_id = query.edited_message.message_id
    
    # 3. Setup global status and add job to queue
    PROCESSING_STATUS[file_id] = {
        'status': 'QUEUED',
        'task_name': task_name,
        'chat_id': query.message.chat_id,
        'progress_message_id': progress_message_id,
        'start_time': time.time(),
        'total_bytes': original_size,
        'input_path': input_path,
        'output_path': output_path,
        'current_step': 0,
        'total_steps': 1, # Updated inside the processing job
    }
    
    job_payload = {
        'file_id': file_id,
        'chat_id': query.message.chat_id,
        'task_name': task_name,
        'input_path': input_path,
        'output_path': output_path,
        'context': context,
    }
    await JOB_QUEUE.put(job_payload)
    logger.info(f"Job added to queue: {file_id} - Queue size: {JOB_QUEUE.qsize()}")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a user-friendly message."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    try:
        if update.effective_message:
            await update.effective_message.reply_text(
                "🚨 An unexpected error occurred. Please try again or send a new file."
            )
    except Exception as e:
        logger.error(f"Failed to send error message to user: {e}")


# --- Main Application Setup ---

# In pdf_magic_bot_complete.py

# ... (Keep post_init and shutdown functions as they are, but perhaps rename shutdown for clarity)

async def post_shutdown(application: Application) -> None:
    """Called when the application is shutting down, gracefully stops tasks."""
    logger.info("Application shutting down. Cancelling job consumer.")
    if 'job_consumer_task' in application.bot_data:
        application.bot_data['job_consumer_task'].cancel()
    logger.info("Shutdown complete.")


def main() -> None:
    """Start the bot."""
    setup_directories()

    # Create the Application and pass your bot's token.
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        # The lifecycle hooks are passed to .build() as Application constructor arguments
        .build(
            post_init=post_init, 
            post_shutdown=post_shutdown # <-- CORRECTED
        )
    )

    # Handlers (rest remains the same)
    application.add_handler(CommandHandler("start", start_handler))
    # ... other handlers ...

    # Run the bot
    logger.info("Bot started. Press Ctrl-C to stop.")
    application.run_polling(poll_interval=1.0, allowed_updates=Update.ALL_TYPES)
    

if __name__ == "__main__":
    main()
