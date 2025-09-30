import os
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, Optional

# --- External Libraries ---
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

# Third-party libraries for PDF/Image processing (must be installed)
# import pdf2image  # Requires poppler
# from PIL import Image
# import cv2        # OpenCV
# import numpy as np
# from PyPDF2 import PdfReader, PdfWriter # Or pypdf for modern usage

# --- Configuration ---
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN_HERE"
LOG_LEVEL = logging.INFO
BASE_DIR = "data"
JOB_QUEUE: asyncio.Queue = asyncio.Queue()  # Global job queue
PROCESSING_STATUS: Dict[str, Dict[str, Any]] = {}  # Status tracking: {file_id: {status, progress_message_id, start_time, total_bytes, ...}}

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

# --- Real-Time Progress Tracking ---

async def progress_tracker(file_id: str, context: ContextTypes.DEFAULT_TYPE):
    """
    Asynchronous function to send real-time progress updates.
    Runs concurrently with the processing job.
    """
    chat_id = PROCESSING_STATUS[file_id]['chat_id']
    message_id = PROCESSING_STATUS[file_id]['progress_message_id']
    total_bytes = PROCESSING_STATUS[file_id]['total_bytes']
    
    # Simulate the total work steps for a process (e.g., number of pages)
    total_steps = PROCESSING_STATUS[file_id].get('total_steps', 10) 
    
    # Convert bytes to MB for display
    total_mb = total_bytes / (1024 * 1024)
    
    while PROCESSING_STATUS[file_id]['status'] in ['QUEUED', 'PROCESSING']:
        start_time = PROCESSING_STATUS[file_id]['start_time']
        elapsed_time = time.time() - start_time
        
        # Get current progress from the processing function (simulated here)
        current_step = PROCESSING_STATUS[file_id].get('current_step', 0)
        
        # Calculate percentage (0 to 100)
        percentage = min(100, int((current_step / total_steps) * 100)) 

        # Speed and ETA calculation (Simplified simulation based on total_mb and elapsed_time)
        # In a real scenario, you'd calculate this based on MB *actually* processed.
        # Here we simulate by assuming work is proportional to time.
        
        if elapsed_time > 0 and percentage > 0:
            simulated_processed_mb = total_mb * (percentage / 100)
            processing_speed_mbps = simulated_processed_mb / elapsed_time
            
            # Simple linear ETA
            if percentage < 100:
                eta_seconds = elapsed_time * (100 - percentage) / percentage
                eta_display = f"{int(eta_seconds)}s"
            else:
                eta_display = "Finishing..."
            
            progress_text = (
                f"⏳ **Status: {PROCESSING_STATUS[file_id]['status']}**\n"
                f"🚀 **Task:** {PROCESSING_STATUS[file_id]['task_name']}\n"
                f"📊 **Progress:** {percentage}%\n"
                f"💾 **Total Size:** {total_mb:.2f} MB\n"
                f"🔄 **Processed (Est):** {simulated_processed_mb:.2f} MB\n"
                f"⚡ **Speed:** {processing_speed_mbps:.2f} MB/s\n"
                f"🕰️ **ETA:** {eta_display}"
            )
        else:
            progress_text = (
                f"⏳ **Status: {PROCESSING_STATUS[file_id]['status']}**\n"
                f"🚀 **Task:** {PROCESSING_STATUS[file_id]['task_name']}\n"
                f"📊 **Progress:** 0%\n"
                f"💾 **Total Size:** {total_mb:.2f} MB\n"
                f"🔄 **Processed (Est):** 0.00 MB\n"
                f"⚡ **Speed:** N/A\n"
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
            # Handle message not modified or message deleted error
            logger.warning(f"Could not update progress message for {file_id}: {e}")
            break
            
        # Update interval
        await asyncio.sleep(2) 
        
    logger.info(f"Progress tracker stopped for {file_id}. Final status: {PROCESSING_STATUS[file_id]['status']}")

# --- Core Processing Logic (Placeholders) ---

# NOTE: In a real-world scenario, these functions would use pdf2image, PIL, and OpenCV
# to implement the actual transformations. They must be non-blocking (async) or run in 
# a thread/process pool to prevent blocking the main asyncio loop. We will use a 
# synchronous blocking function wrapper here.

def update_processing_progress(file_id: str, current_step: int):
    """Safely update the progress step."""
    if file_id in PROCESSING_STATUS:
        PROCESSING_STATUS[file_id]['current_step'] = current_step

async def process_pdf_job(
    input_path: str, output_path: str, task_name: str, file_id: str, context: ContextTypes.DEFAULT_TYPE
) -> Optional[str]:
    """
    Main asynchronous job function. Runs the heavy-lifting logic in a separate
    thread pool to avoid blocking the event loop.
    
    Returns: The path to the processed output file, or None on failure.
    """
    logger.info(f"Starting {task_name} job for {file_id}...")
    
    total_pages = 5 # Placeholder: Use PyPDF2 to get the actual page count
    PROCESSING_STATUS[file_id]['total_steps'] = total_pages # One step per page
    
    def blocking_processing():
        """This function runs in a separate thread."""
        try:
            # Simulate the time-consuming process (e.g., page-by-page conversion and processing)
            for i in range(1, total_pages + 1):
                logger.info(f"Processing page {i}/{total_pages} for {file_id}")
                
                # --- ACTUAL PROCESSING LOGIC GOES HERE ---
                # 1. Convert page 'i' of PDF to image using pdf2image.
                # 2. Apply OpenCV/PIL transformations based on task_name.
                #    - SCAN: cv2.getPerspectiveTransform, cv2.adaptiveThreshold
                #    - UPSCALE: PIL.Image.resize (high-quality), enhance contrast
                #    - COMPRESS: Reduce image quality before re-assembling, or use PyPDF2 to compress streams.
                # 3. Save the processed image/PDF fragment.
                
                time.sleep(1 + i * 0.5) # Simulate processing time
                update_processing_progress(file_id, i) # Update progress step

            # Final step: Re-assemble pages into a new PDF
            time.sleep(2) 
            
            # Placeholder: Create a dummy output file for demonstration
            with open(output_path, 'w') as f:
                f.write(f"Processed PDF content for {task_name}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error during blocking process for {file_id}: {e}")
            return None

    # Run the blocking function in a separate thread
    loop = asyncio.get_running_loop()
    # The default executor for run_in_executor is a ThreadPoolExecutor
    final_output_path = await loop.run_in_executor(None, blocking_processing) 
    
    return final_output_path


# --- Job Queue Management ---

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
                    read_timeout=60, # Increase timeout for large files
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
            
            # Inform the user of the error
            error_message = f"❌ **Error during {task_name}**:\n"
            if 'Unsupported format' in str(e):
                 error_message += "The file format might be unsupported or corrupt."
            elif 'very large' in str(e):
                 error_message += "The file is too large or complex for the current server capacity."
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
            
            # Cleanup - optional: auto-delete processed files
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if final_output_path and os.path.exists(final_output_path):
                    # In a production bot, you might want to wait longer before deleting
                    # or have a separate cleanup task. Deleting immediately is safer for storage.
                    os.remove(final_output_path)
            except Exception as e:
                logger.warning(f"Could not delete files for {file_id}: {e}")
            
            del PROCESSING_STATUS[file_id] # Remove tracking status
            JOB_QUEUE.task_done() # Signal that the job is done
            logger.info(f"Job {file_id} finished. Queue size: {JOB_QUEUE.qsize()}")


# --- Telegram Handlers ---

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and instructions."""
    await update.message.reply_text(
        "👋 Welcome to PDF Magic Bot!\n\n"
        "Please send me a **PDF file** to get started. I can scan, upscale, or compress it."
    )

async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming document messages."""
    document = update.message.document
    
    # Check if the document is a PDF
    if not document.mime_type or 'application/pdf' not in document.mime_type:
        await update.message.reply_text(
            "🚫 I can only process **PDF files** for now. Please send a valid PDF."
        )
        return

    # Check file size (Telegram limit is 50MB, but we might want a lower limit for processing)
    if document.file_size > 50 * 1024 * 1024: # 50 MB
        await update.message.reply_text(
            "⚠️ File too large. The current processing limit is 50MB. Please use a smaller file."
        )
        return

    # Store file information temporarily for callback
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
    await query.answer() # Acknowledge the button press
    
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
    
    # Determine the job name and output suffix
    if task_action == "process_scan":
        task_name = "Scan & Clean"
        suffix = "_scanned.pdf"
    elif task_action == "process_upscale":
        task_name = "Upscale & Enhance"
        suffix = "_enhanced.pdf"
    elif task_action == "process_compress":
        task_name = "Compress"
        suffix = "_compressed.pdf"
    else:
        await query.edit_message_text("🚫 Invalid option selected.")
        return

    # 1. Initiate download and setup paths
    downloaded_file = await context.bot.get_file(file_id)
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(BASE_DIR, 'input', f"{unique_id}_original.pdf")
    # Clean up name for output
    base_name = os.path.splitext(original_name)[0]
    output_path = os.path.join(BASE_DIR, 'output', f"{base_name}{suffix}")
    
    try:
        # NOTE: Using download_to_drive is synchronous, which is acceptable 
        # for a quick download before the main heavy job starts.
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
    # Get the message ID for real-time updates
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
        'total_steps': 1, # Will be updated in the job worker
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
        # A bit generic, but safe for any unexpected error
        if update.effective_message:
            await update.effective_message.reply_text(
                "🚨 An unexpected error occurred. Please try again or send a new file."
            )
    except Exception as e:
        logger.error(f"Failed to send error message to user: {e}")


# --- Main Application Setup ---

async def post_init(application: Application) -> None:
    """Called when the application has started, sets up background tasks."""
    logger.info("Application started. Setting up job consumer.")
    # Start the job consumer worker as a permanent background task
    application.bot_data['job_consumer_task'] = application.create_task(job_consumer(application))

async def shutdown(application: Application) -> None:
    """Called when the application is shutting down, gracefully stops tasks."""
    logger.info("Application shutting down. Cancelling job consumer.")
    # Cancel the job consumer task
    if 'job_consumer_task' in application.bot_data:
        application.bot_data['job_consumer_task'].cancel()
    
    # Wait for the queue to empty (optional, but good practice)
    # await JOB_QUEUE.join() 
    logger.info("Shutdown complete.")


def main() -> None:
    """Start the bot."""
    setup_directories()

    # Create the Application and pass your bot's token.
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .shutdown(shutdown)
        .build()
    )

    # Handlers
    application.add_handler(CommandHandler("start", start_handler))
    
    # Filter for documents that are NOT photos, videos, or audio (i.e., general files)
    # Further PDF check is done in the handler.
    application.add_handler(MessageHandler(filters.Document.ALL & ~filters.ChatType.CHANNEL, document_handler))
    
    application.add_handler(CallbackQueryHandler(callback_handler))
    
    # Error Handler
    application.add_error_handler(error_handler)

    # Run the bot until the user presses Ctrl-C
    # In a production environment, you would use a web hook for deployment (e.g., Heroku/Cloud Run)
    logger.info("Bot started. Press Ctrl-C to stop.")
    application.run_polling(poll_interval=1.0, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
