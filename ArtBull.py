#!/usr/bin/env python
"""
Discord Bot with Slash Commands:
1) /gpt            - Get a text reply from OpenAI's o1-preview model.
2) /dalle          - Generate an image using DALL‑E 3 (sends file only, no URLs).
3) /ollama         - Get a reply from a locally hosted LLM (e.g., via an Ollama API).
4) /sd             - Generate an image using a locally hosted Stable Diffusion API.
5) /toggle_mentions - Toggle the bot's ability to respond automatically when mentioned.
6) /flux           - Generate an image using the Flux Pro API.
   
All command settings (enable/disable, name, and description) can be configured via COMMAND_CONFIG.
Additionally, the number of prior messages to include when building context can be set via HISTORY_LIMIT.
"""

import os
import discord
from discord import app_commands
from discord.ext import commands
from typing import Optional, Dict, Any
import openai
import requests
import asyncio
import base64
import io
import logging
import json
import re
import time
from functools import wraps

# -------------------- Message Splitting Helpers --------------------
def split_message(text: str, max_length: int = 1990) -> list:
    """
    Splits the given text into chunks that are no longer than max_length characters.
    It first attempts to split by newline, but if a single line is too long,
    it splits arbitrarily.
    
    Using 1990 as default to provide a safety margin for Discord's 2000 char limit.
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    for line in text.splitlines():
        # Add a newline if this isn't the first line
        if current_chunk and len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk.rstrip())
            current_chunk = line + "\n"
        elif not current_chunk and len(line) > max_length:
            # Line is too long, need to split within the line
            while len(line) > max_length:
                chunks.append(line[:max_length])
                line = line[max_length:]
            current_chunk = line + "\n" if line else ""
        else:
            current_chunk += line + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.rstrip())
    
    return chunks

async def send_split_message_channel(channel: discord.TextChannel, text: str):
    """
    Splits the given text into chunks and sends each chunk to the specified channel.
    """
    chunks = split_message(text)
    for chunk in chunks:
        await channel.send(chunk)

async def send_split_message(message: discord.Message, text: str):
    """
    Splits the text into chunks using split_message() and sends each chunk as a reply.
    """
    chunks = split_message(text)
    first_reply = None
    for i, chunk in enumerate(chunks):
        try:
            if i == 0:
                first_reply = await message.reply(chunk)
            else:
                await message.channel.send(chunk)
        except Exception as e:
            logger.error(f"Error sending chunk {i}: {e}")
            try:
                await message.channel.send(f"Error sending message chunk: {e}")
            except:
                pass
    return first_reply

async def update_status_message(msg: discord.Message, start_time: float, stop_event: asyncio.Event):
    """Continuously update the status message with elapsed time until stop_event is set."""
    try:
        while not stop_event.is_set():
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            plural = "s" if elapsed != 1 else ""
            await msg.edit(content=f"Working on your request for {elapsed} second{plural}...")
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    except discord.errors.NotFound:
        # Message was deleted or not found
        pass
    except Exception as e:
        logger.error(f"Error updating status message: {e}")

# -------------------- Configuration --------------------

# Discord and OpenAI credentials
DISCORD_TOKEN = ""
OPENAI_API_KEY = ""

# API configuration / Endpoints
OpenAI_MODEL = "o1-preview" #Ensure to use a model that your tier has access to.
DALL_E3_MODEL = "dall-e-3"
LOCAL_LLM_API_URL = "http://127.0.0.1/api/generate"
STABLE_DIFFUSION_API_URL = "http://127.0.0.1:8000/api/generate"

# Flux API configuration
FLUX_API_KEY = ""
FLUX_POST_URL = "https://api.us1.bfl.ai/v1/flux-pro-1.1-ultra"
FLUX_GET_URL = "https://api.us1.bfl.ai/v1/get_result"

# Additional model configuration
OLLAMA_MODEL = ""

# Command configuration: enable/disable, rename, and description
COMMAND_CONFIG = {
    "openai": {
        "enabled": True,
        "name": "gpt",
        "description": "Get a reply from OpenAI's o1-preview model (This can get expensive.)"
    },
    "dalle": {
        "enabled": True,
        "name": "dalle",
        "description": "Generate an image with DALL-E-3"
    },
    "ollama": {
        "enabled": True,
        "name": "ollama",
        "description": "Get a reply from a local LLM (Ollama API)"
    },
    "stable_diffusion": {
        "enabled": True,
        "name": "sd",
        "description": "Generate an image using locally hosted Stable Diffusion"
    },
    "fluxart": {
        "enabled": True,
        "name": "flux",
        "description": "Generate an image using the Flux 1.1 Pro API"
    }
}

# Additional configuration for mentions
HISTORY_LIMIT = 5            # Number of prior messages to include as context
respond_to_mentions = True   # Global toggle for @mention responses

# System message for @mention API calls
MENTION_SYSTEM_MESSAGE = (
    "This API call is initiated by a Discord user. "
    "Please note that Discord uses Markdown for formatting: "
    "use *italic*, **bold**, __underline__, and `inline code` as appropriate. "
    "Format the response in plain text suitable for Discord messages."
)

# Request timeouts
DEFAULT_TIMEOUT = 30  # seconds
FLUX_POLL_INTERVAL = 2  # seconds

# Placeholder image configuration
PLACEHOLDER_IMAGE_URL = ""  # Set to a URL for a custom placeholder image for image gen commands to have something to replace.
MAX_MESSAGE_LENGTH = 1990   # Slightly under Discord's 2000 character limit

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for less verbose output
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
active_tasks = {}  # Global dictionary to keep track of long-running tasks

# -------------------- Helper Functions --------------------
async def download_image_to_file(url: str, filename: str) -> discord.File:
    """Download an image from a URL and return it as a Discord File object"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=DEFAULT_TIMEOUT) as response:
            if response.status != 200:
                raise Exception(f"Failed to download image: HTTP {response.status}")
            image_data = await response.read()
            return discord.File(fp=io.BytesIO(image_data), filename=filename)

def get_placeholder_image():
    """Returns either a custom placeholder image or the default tiny image"""
    if PLACEHOLDER_IMAGE_URL:
        try:
            response = requests.get(PLACEHOLDER_IMAGE_URL, timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return io.BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error loading custom placeholder image: {e}")
    
    # Fall back to the built-in tiny image
    return io.BytesIO(
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        b'\x00\x00\x00\x10\x00\x00\x00\x10\x08\x06\x00\x00\x00'
        b'\x1f\xf3\xff\x61\x00\x00\x00\x19tEXtSoftware\x00'
        b'Adobe ImageReadyq\xc9e<\x00\x00\x01\x19IDATx\xda\xa4\x93M'
        b'\n\xc20\x10\x85\xb7m\xa1T\xbb\x10\xf7\x96\x8a[\xef\xe5\x05'
        b'\xbc\x88\xa2\xb8\xf6\x1e\n\xfe\xac\x05\x11\xf7\x82\xe2?'
        b'\x16\xb4I\x9a\xc6D\x93\x14\x0bE\x07\x86I&\x99\xbc\xd7\xc9'
        b'\x0c\nM\x8cy\x9b\x0c\x0b\xfcr\x9f\xa1\xf8\x857lp$\xd0'
        b'\xe3\xba\xe7\x1d*\x14\xa4\x06\xbb\xfeP\xc4h\xbfN\\\xcc'
        b'\x17\xa2\x06\x0f\xd84a\x80g+\x1a\xa7\xeb\x87\xc7r\x950'
        b'\x99\t\xe5\xc9\x0b\xe6\x8b\x99\xc8=6B\xc3\x0c\xe6\x908'
        b'\xbe\xc6E\x1c\x91{<\xd0_\x83\x0b\xec\x91\xd5\x90K\xec'
        b'\xcfz\xa4\x03N`\xb9\xa2\x06\x05n\xf0\x88|\\\x01\x7f\xa0'
        b'\xb7\x06\x0e6cZo\xb1jx\xf1\x86G\xd8\x95\xaf\xe5\xc9u'
        b'\x80=\xce\x905\xb8g\x12\xcb]Xb\x8a\r\xe5\xecZl)\x8a'
        b'\xe4\x14\x05\x94\x81\xb2\x90\xcdL\xa0^\xaf\xe11\x83(\x08'
        b'\xe4\xa6c\xed\xc9\xb4\x8d\x92\xb2O\x81\x9e\xd6F\x1d\x05'
        b'\x81\xcc\xc5F\x1a\x9b\xf8\xbez,7m\xe0\xf7\x9fyp\x16'
        b'\xa8\xf6\n\xdfJ\xfa\x13\x03\x93\x9f\xe1\'\xc7/\x01\x06\x00'
        b'\x7f\xe4`\xe6\xb9%\x06s\x00\x00\x00\x00IEND\xaeB`\x82'
    )

def error_handler(func):
    """Decorator for handling errors in slash commands"""
    @wraps(func)
    async def wrapper(interaction: discord.Interaction, *args, **kwargs):
        try:
            await func(interaction, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}", exc_info=True)
            error_message = f"Error: {str(e)}"
            if interaction.response.is_done():
                await interaction.followup.send(error_message)
            else:
                await interaction.response.send_message(error_message, ephemeral=True)
    return wrapper

# -------------------- Discord Bot Setup --------------------
import aiohttp
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!?!?!", intents=intents)

@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}!")
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} command(s).")
    except Exception as e:
        logger.error("Error syncing commands", exc_info=True)

# -------------------- Slash Commands --------------------

# 1) /gpt Command (with message splitting for long text responses)
if COMMAND_CONFIG["openai"]["enabled"]:
    @bot.tree.command(name=COMMAND_CONFIG["openai"]["name"],
                      description=COMMAND_CONFIG["openai"]["description"])
    @app_commands.describe(
        prompt="The prompt for GPT.",
        model="Optional model override. Defaults to o1-preview."
    )
    async def gpt(interaction: discord.Interaction, prompt: str, model: Optional[str] = OpenAI_MODEL):
        logger.info(f"/gpt called with prompt: {prompt[:50]}... and model: {model}")
        
        await interaction.response.send_message("Working on your request...")
        status_msg = await interaction.original_response()
        
        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[f"gpt_{interaction.id}"] = update_task
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            if model == "o1-preview":
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=15000
                )
            else:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=15000
                )
            
            reply = response.choices[0].message.content.strip()
            if not reply:
                reply = "Received an empty response from the model."
        except Exception as e:
            logger.error("Error in /gpt command", exc_info=True)
            reply = f"Error: {str(e)}"
        finally:
            stop_update_event.set()
            await asyncio.sleep(0.5)
            # Prepare the full response with a prefix
            prefix = f"Here's your expensive af response <@{interaction.user.id}>\n\n"
            full_text = prefix + reply
            if len(full_text) <= MAX_MESSAGE_LENGTH:
                await interaction.edit_original_response(content=full_text)
            else:
                chunks = split_message(reply)
                first_chunk = prefix + chunks[0]
                if len(first_chunk) > MAX_MESSAGE_LENGTH:
                    # If even with the first chunk it's too long, split differently
                    first_chunk = prefix
                    await interaction.edit_original_response(content=first_chunk)
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.edit_original_response(content=first_chunk)
                    for extra in chunks[1:]:
                        await interaction.followup.send(extra)
            active_tasks.pop(f"gpt_{interaction.id}", None)

# 2) /dalle Command remains unchanged (image response)
if COMMAND_CONFIG["dalle"]["enabled"]:
    @bot.tree.command(name=COMMAND_CONFIG["dalle"]["name"],
                      description=COMMAND_CONFIG["dalle"]["description"])
    async def dalle(interaction: discord.Interaction, prompt: str):
        logger.info(f"/dalle called with prompt: {prompt[:50]}...")
        
        # Use the placeholder image function
        placeholder = get_placeholder_image()
        await interaction.response.send_message("Generating your image...", file=discord.File(placeholder, filename="placeholder.png"))
        status_msg = await interaction.original_response()
        
        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[f"dalle_{interaction.id}"] = update_task
        
        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            response = await asyncio.to_thread(
                client.images.generate,
                model=DALL_E3_MODEL,
                prompt=prompt,
                n=1,
                size="1024x1024",
                response_format="b64_json"
            )
            
            if not response.data:
                raise ValueError("No image data returned from DALL‑E 3 API.")
                
            data = response.data[0]
            
            if data.b64_json:
                b64_image = data.b64_json
                image_data = base64.b64decode(b64_image)
            elif data.url:
                async with aiohttp.ClientSession() as session:
                    async with session.get(data.url, timeout=DEFAULT_TIMEOUT) as resp:
                        if resp.status != 200:
                            raise Exception(f"Failed to download image: HTTP {resp.status}")
                        image_data = await resp.read()
            else:
                raise ValueError("No image data or URL returned from DALL‑E 3 API.")
                
            file = discord.File(fp=io.BytesIO(image_data), filename="dalle3.png")
            
        except Exception as e:
            logger.error("Error in /dalle command", exc_info=True)
            stop_update_event.set()
            await asyncio.sleep(0.5)
            await interaction.edit_original_response(content=f"Error: {str(e)}")
            active_tasks.pop(f"dalle_{interaction.id}", None)
            return
        finally:
            stop_update_event.set()
            await asyncio.sleep(0.5)
        
        final_content = f"Here's your $0.04 image <@{interaction.user.id}>"
        await interaction.edit_original_response(content=final_content, attachments=[file])
        active_tasks.pop(f"dalle_{interaction.id}", None)

# 3) /ollama Command (with message splitting for long text responses)
if COMMAND_CONFIG["ollama"]["enabled"]:
    @bot.tree.command(name=COMMAND_CONFIG["ollama"]["name"],
                      description=COMMAND_CONFIG["ollama"]["description"])
    async def ollama(interaction: discord.Interaction, prompt: str):
        logger.info(f"/ollama called with prompt: {prompt[:50]}...")
        
        await interaction.response.send_message("Consulting the local model...")
        status_msg = await interaction.original_response()
        
        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[f"ollama_{interaction.id}"] = update_task
        
        try:
            payload = {"prompt": prompt, "model": OLLAMA_MODEL}
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LOCAL_LLM_API_URL, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1200)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API returned status code {response.status}")
                    
                    text = await response.text()
                    lines = text.strip().splitlines()
                    
            reply_parts = []
            for line in lines:
                try:
                    obj = json.loads(line)
                    if "response" in obj:
                        reply_parts.append(obj["response"])
                    elif "reply" in obj:
                        reply_parts.append(obj["reply"])
                except Exception:
                    continue
                    
            reply = " ".join(reply_parts).strip()
            reply = re.sub(r'\s+([,.?!])', r'\1', reply)
            if not reply:
                reply = "No reply received from the local LLM."
                
        except Exception as e:
            logger.error("Error in /ollama command", exc_info=True)
            reply = f"Error: {str(e)}"
        finally:
            stop_update_event.set()
            await asyncio.sleep(0.5)
            prefix = f"Here's your long awaited response <@{interaction.user.id}>\n\n"
            full_text = prefix + reply
            if len(full_text) <= MAX_MESSAGE_LENGTH:
                await interaction.edit_original_response(content=full_text)
            else:
                chunks = split_message(reply)
                first_chunk = prefix + chunks[0]
                if len(first_chunk) > MAX_MESSAGE_LENGTH:
                    # If even with the first chunk it's too long, split differently
                    first_chunk = prefix
                    await interaction.edit_original_response(content=first_chunk)
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.edit_original_response(content=first_chunk)
                    for extra in chunks[1:]:
                        await interaction.followup.send(extra)
            active_tasks.pop(f"ollama_{interaction.id}", None)

# 4) /stable_diffusion Command
if COMMAND_CONFIG["stable_diffusion"]["enabled"]:
    @bot.tree.command(name=COMMAND_CONFIG["stable_diffusion"]["name"],
                      description=COMMAND_CONFIG["stable_diffusion"]["description"])
    @app_commands.describe(
        prompt="The text prompt to generate an image from.",
        inference_steps="Number of inference steps (default: 10).",
        image_width="Width of the generated image in pixels (default: 1024).",
        image_height="Height of the generated image in pixels (default: 1024).",
        number_of_images="Number of images to generate (default: 1).",
        sampler_index="Sampler algorithm to use (default: Euler).",
        seed="Optional seed for reproducible results."
    )
    async def stable_diffusion(
        interaction: discord.Interaction,
        prompt: str,
        inference_steps: int = 10,
        image_width: int = 1024,
        image_height: int = 1024,
        number_of_images: int = 1,
        sampler_index: str = "Euler",
        seed: Optional[int] = None
    ):
        logger.info(f"/stable_diffusion called with prompt: {prompt[:50]}...")
        
        # Use the placeholder image function
        placeholder = get_placeholder_image()
        await interaction.response.send_message("Generating image with Stable Diffusion...", file=discord.File(placeholder, filename="placeholder.png"))
        status_msg = await interaction.original_response()
        
        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[f"sd_{interaction.id}"] = update_task
        
        try:
            payload = {
                "prompt": prompt,
                "inference_steps": inference_steps,
                "image_width": min(max(image_width, 256), 2048),
                "image_height": min(max(image_height, 256), 2048),
                "number_of_images": min(number_of_images, 4),
                "sampler_index": sampler_index
            }
            if seed is not None:
                payload["seed"] = seed

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    STABLE_DIFFUSION_API_URL, 
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=1200)
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API returned status code {response.status}")
                    
                    data = await response.json()
            
            images = data.get("images", [])
            if not images:
                raise Exception("No images returned from Stable Diffusion API")
                
            img_b64 = images[0]
            image_data = base64.b64decode(img_b64)
            file = discord.File(fp=io.BytesIO(image_data), filename="stable_diffusion.png")
            
        except Exception as e:
            logger.error("Error in /stable_diffusion command", exc_info=True)
            stop_update_event.set()
            await asyncio.sleep(0.5)
            await interaction.edit_original_response(content=f"Error: {str(e)}")
            active_tasks.pop(f"sd_{interaction.id}", None)
            return
        finally:
            stop_update_event.set()
            await asyncio.sleep(0.5)
            
        final_content = f"Here's your long awaited image <@{interaction.user.id}>"
        await interaction.edit_original_response(content=final_content, attachments=[file])
        active_tasks.pop(f"sd_{interaction.id}", None)

# 5) /toggle_mentions Command
@bot.tree.command(name="toggle_mentions", description="Toggle the bot's ability to respond to @mentions with context.")
async def toggle_mentions(interaction: discord.Interaction):
    global respond_to_mentions
    respond_to_mentions = not respond_to_mentions
    status = "enabled" if respond_to_mentions else "disabled"
    logger.info(f"@mention responses have been {status}.")
    await interaction.response.send_message(f"<@{interaction.user.id}> used /toggle_mentions\n@mention responses are now {status}.", ephemeral=True)

# 6) /fluxart Command
if COMMAND_CONFIG["fluxart"]["enabled"]:
    @bot.tree.command(name=COMMAND_CONFIG["fluxart"]["name"],
                      description=COMMAND_CONFIG["fluxart"]["description"])
    @app_commands.describe(
        prompt="The art prompt for Flux image generation.",
        aspect_ratio="Aspect ratio of the image between 21:9 and 9:21 (default 1:1)",
        output_format="Output format: png or jpeg (default png)",
        prompt_upsampling="Enable prompt upsampling? (default false)",
        raw="Enable raw mode (less processed images)? (default false)",
        seed="Optional seed for reproducibility (default random)",
        safety_tolerance="Safety tolerance (0-6, default 6)"
    )
    async def fluxart(
        interaction: discord.Interaction,
        prompt: str,
        aspect_ratio: str = "1:1",
        output_format: str = "png",
        prompt_upsampling: bool = False,
        raw: bool = False,
        seed: Optional[int] = None,
        safety_tolerance: int = 6
    ):
        logger.info(f"/fluxart called with prompt: {prompt[:50]}...")
        
        safety_tolerance = min(max(safety_tolerance, 0), 6)
        if output_format.lower() not in ["jpeg", "png"]:
            output_format = "png"
        
        # Use the placeholder image function
        placeholder = get_placeholder_image()
        await interaction.response.send_message("Sending request to Flux API...", file=discord.File(placeholder, filename="placeholder.png"))
        status_msg = await interaction.original_response()
        
        data = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format.lower(),
            "prompt_upsampling": prompt_upsampling,
            "raw": raw,
            "safety_tolerance": safety_tolerance
        }
        if seed is not None:
            data["seed"] = seed

        headers = {
            "x-key": FLUX_API_KEY,
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[f"flux_{interaction.id}"] = update_task
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    FLUX_POST_URL, 
                    json=data, 
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT
                ) as response:
                    if response.status != 200:
                        raise Exception(f"API returned status code {response.status}")
                    
                    post_data = await response.json()
                    
            request_id = post_data.get("id")
            if not request_id:
                raise Exception("No request ID received from Flux API.")
                
            await status_msg.edit(content="Request sent. Waiting for image generation...")
            
            poll_headers = {
                "x-key": FLUX_API_KEY,
                "accept": "application/json"
            }
            
            image_url = None
            max_polls = 60
            polls = 0
            
            while polls < max_polls:
                polls += 1
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{FLUX_GET_URL}?id={request_id}",
                        headers=poll_headers,
                        timeout=DEFAULT_TIMEOUT
                    ) as response:
                        if response.status != 200:
                            if polls >= max_polls:
                                raise Exception(f"Error polling Flux API: HTTP {response.status}")
                            await asyncio.sleep(FLUX_POLL_INTERVAL)
                            continue
                        
                        get_data = await response.json()
                
                status = get_data.get("status", "Unknown")
                if status == "Ready":
                    result_data = get_data.get("result", {})
                    image_url = result_data.get("sample")
                    if image_url:
                        break
                    else:
                        raise Exception("Flux API returned 'Ready' but no image URL found.")
                elif status in ["Content Moderated", "Request Moderated"]:
                    moderation_reasons = get_data.get("details", {}).get("Moderation Reasons", ["Unknown reason"])
                    raise Exception(f"Request moderated: {', '.join(moderation_reasons)}")
                elif status in ["Failed", "Error"]:
                    raise Exception(f"Flux API returned error status: {status}")
                
                await asyncio.sleep(FLUX_POLL_INTERVAL)
            
            if not image_url:
                raise Exception("Timed out waiting for Flux to generate image")
                
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=DEFAULT_TIMEOUT) as response:
                    if response.status != 200:
                        raise Exception(f"Failed to download image: HTTP {response.status}")
                    image_bytes = await response.read()
            
            sanitized_prompt = "".join(c for c in prompt if c.isalnum() or c in (" ", "_")).rstrip()[:20]
            filename = f"FluxArt_{sanitized_prompt}_{int(time.time())}.{output_format}"
            file = discord.File(fp=io.BytesIO(image_bytes), filename=filename)
            
        except Exception as e:
            logger.error("Error in /fluxart command", exc_info=True)
            stop_update_event.set()
            await asyncio.sleep(0.5)
            await interaction.edit_original_response(content=f"Error: {str(e)}")
            active_tasks.pop(f"flux_{interaction.id}", None)
            return
        finally:
            stop_update_event.set()
            await asyncio.sleep(0.5)
            
        final_content = f"Here's your $0.06 image <@{interaction.user.id}>"
        await interaction.edit_original_response(content=final_content, attachments=[file])
        active_tasks.pop(f"flux_{interaction.id}", None)

# -------------------- Message Listener for @mentions --------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    if respond_to_mentions and (bot.user in message.mentions):
        logger.info(f"Bot was mentioned in message {message.id}. Gathering conversation history...")
        
        content = message.content
        for mention in message.mentions:
            content = content.replace(f'<@{mention.id}>', f'@{mention.display_name}')
        
        history_messages = []
        fetched_messages = []
        
        async for msg in message.channel.history(limit=HISTORY_LIMIT + 1):
            if msg.id != message.id:
                fetched_messages.append(msg)
        
        for msg in reversed(fetched_messages):
            sender = msg.author.display_name
            clean_content = msg.content
            for mention in msg.mentions:
                clean_content = clean_content.replace(f'<@{mention.id}>', f'@{mention.display_name}')
            if not msg.author.bot:
                history_messages.append(f"{sender}: {clean_content}")
        
        history_messages.append(f"{message.author.display_name}: {content}")
        context_prompt = "\n".join(history_messages)
        logger.debug(f"Constructed context prompt:\n{context_prompt}")

        start_time = asyncio.get_event_loop().time()
        stop_update_event = asyncio.Event()
        status_msg = await message.channel.send("Working on your request for 0 seconds...")
        
        task_id = f"mention_{message.id}"
        update_task = asyncio.create_task(update_status_message(status_msg, start_time, stop_update_event))
        active_tasks[task_id] = update_task

        try:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
            messages_payload = [
                {"role": "system", "content": MENTION_SYSTEM_MESSAGE},
                {"role": "user", "content": context_prompt}
            ]
    
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=messages_payload,
                max_tokens=4500,
                temperature=1
            )
    
            reply = response.choices[0].message.content.strip()
            if not reply:
                reply = "Received an empty response from the model."
            logger.debug(f"@mention response: {reply}")
    
        except Exception as e:
            logger.error("Error responding to @mention", exc_info=True)
            reply = f"Error: {e}"
        
        stop_update_event.set()
        await asyncio.sleep(0.5)
        active_tasks.pop(task_id, None)
        
        if len(reply) > MAX_MESSAGE_LENGTH:
            await status_msg.delete()
            await send_split_message_channel(message.channel, reply)
        else:
            try:
                await status_msg.edit(content=reply)
            except discord.errors.NotFound:
                await message.channel.send(reply)

    await bot.process_commands(message)

# -------------------- Main Entry Point --------------------
if __name__ == "__main__":
    print(f"OpenAI library version: {openai.__version__}")
    logger.info("Starting Discord bot...")
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        logger.critical("Bot encountered a critical error", exc_info=True)
        print(f"CRITICAL ERROR: {e}")