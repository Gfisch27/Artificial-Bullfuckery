# ArtBull Discord Bot

## This single-file bot was written 99.9% by AI. I take 0 credit for being able to write this myself. I have 0 programming skill. (I just enjoyed the process and thought the result was worth sharing.)

A versatile Discord bot that provides text and image generation capabilities using various AI models.

**I did not follow best practice and use a .env file the API secrets. I didn't want to do that purely so it can be a single file.

![Discord Bot](https://img.shields.io/badge/discord-bot-5865F2?style=for-the-badge&logo=discord&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Features

### Text Generation
- **GPT (o1-preview)**: Generate text responses using OpenAI's latest models
- **Ollama**: Get responses from a locally hosted LLM via Ollama API

### Image Generation
- **DALL-E 3**: Generate images with OpenAI's DALL-E 3
- **Stable Diffusion**: Create images using a locally hosted Stable Diffusion API
- **Flux Pro**: Generate high-quality art with the Flux API

### Discord Integration
- **Slash Commands**: Easy-to-use slash commands for all features
- **@Mentions**: Have the bot respond when mentioned with context from previous messages
- **Status Updates**: Real-time updates showing elapsed time during processing
- **Message Splitting**: Properly handles long responses by splitting them into multiple messages

## Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/gpt` | Generate text with OpenAI's o1-preview model | `/gpt prompt:Write a story about a space cat` |
| `/dalle` | Generate an image with DALL-E 3 | `/dalle prompt:A cat astronaut floating in space` |
| `/ollama` | Get a response from a local LLM | `/ollama prompt:Explain quantum computing` |
| `/sd` | Create an image with Stable Diffusion | `/sd prompt:Mountain landscape at sunset` |
| `/flux` | Generate art with Flux Pro API | `/flux prompt:Abstract digital art with vibrant colors` |
| `/toggle_mentions` | Toggle the bot's ability to respond to @mentions | `/toggle_mentions` |

## Setup

### Prerequisites
- Python 3.9+
- A Discord account and bot token ([Discord Developer Portal](https://discord.com/developers/applications))
- OpenAI API key (for GPT and DALL-E)
- Flux API key (for Flux art generation)
- Ollama API running locally (optional)
- Stable Diffusion API running locally (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Gfisch27/ArtBull.git
   cd ArtBull
   ```

2. Install dependencies:
   ```bash
   pip install discord.py openai requests
   ```

3. Configure your API keys:
   Edit the following variables in `ArtBull.py`:
   ```python
   DISCORD_TOKEN = "your_discord_token"
   OPENAI_API_KEY = "your_openai_api_key"
   FLUX_API_KEY = "your_flux_api_key"
   ```

4. Configure API endpoints:
   If using local Ollama or Stable Diffusion, update the API URLs:
   ```python
   LOCAL_LLM_API_URL = "http://localhost:11434/api/generate"
   STABLE_DIFFUSION_API_URL = "http://localhost:8000/api/generate"
   ```

5. (Optional) Configure a custom placeholder image:
   ```python
   PLACEHOLDER_IMAGE_URL = "https://example.com/your-placeholder-image.png"
   ```

6. Run the bot:
   ```bash
   python ArtBull.py
   ```

## Configuration Options

You can customize various aspects of the bot by editing the `COMMAND_CONFIG` dictionary in the code:

```python
COMMAND_CONFIG = {
    "o3mini": {
        "enabled": True,
        "name": "gpt",
        "description": "Get a reply from OpenAI's o1-preview model"
    },
    # ... other commands
}
```

You can:
- Enable/disable specific commands
- Rename commands
- Change command descriptions

Other configuration options:
- `HISTORY_LIMIT`: Number of prior messages to include as context for @mentions (default: 5)
- `respond_to_mentions`: Toggle for @mention responses (default: True)
- `MAX_MESSAGE_LENGTH`: Character limit for messages (default: 1990)

## Troubleshooting

- **API Connection Issues**: Ensure your API keys are valid and your local services are running
- **Discord Integration Problems**: Verify that your bot has the necessary permissions in Discord
- **Command Not Found**: Check if the command is enabled in the `COMMAND_CONFIG`
- **Large Messages**: The bot automatically splits large messages, but in very rare cases, you might need to increase `MAX_MESSAGE_LENGTH`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Discord.py](https://github.com/Rapptz/discord.py)
- [OpenAI API](https://openai.com/blog/openai-api)
- [Flux AI](https://flux.ai)
- [Ollama](https://ollama.ai)
