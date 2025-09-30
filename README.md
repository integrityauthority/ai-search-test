# AI Search Test Suite

This script tests and compares the search and web grounding capabilities of various AI models including Gemini, Perplexity, Claude, OpenAI GPT, and Grok.

## Features

- Compare multiple AI models with web search capabilities
- Structured JSON output for consistent result formatting
- Cost calculation and tracking for API usage
- Excel export of results with top-3 cheapest offers per model
- Comprehensive logging and error handling

## Supported Models

- **Google Gemini**: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro
- **Perplexity**: sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro, sonar-deep-research
- **Anthropic Claude**: claude-opus-4, claude-sonnet-4, claude-3-7-sonnet, claude-3-5-haiku
- **OpenAI GPT**: gpt-5, gpt-5-mini, gpt-5-nano
- **xAI Grok**: grok-4, grok-4-fast-reasoning, grok-4-fast-non-reasoning

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Dependencies

For Excel export functionality:
```bash
pip install openpyxl
```

For Grok/xAI API support:
```bash
pip install xai-sdk
```

## Setup

1. Create a `.env` file in the project root
2. Add your API keys (only for the providers you want to use):

```bash
# Google Gemini API (required for Gemini models)
GEMINI_API_KEY=your_gemini_api_key_here

# Perplexity API (required for Perplexity models)
PERPLEXITY_KEY=your_perplexity_api_key_here

# Anthropic Claude API (required for Anthropic models)
ANTHROPIC_KEY=your_anthropic_api_key_here

# OpenAI API (required for OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# xAI/Grok API (required for Grok models)
XAI_API_KEY=your_xai_api_key_here
# or
GROK_API_KEY=your_grok_api_key_here
```

## Usage

Run the script:
```bash
python aisearchtest.py
```

The script will:
1. Load enabled models based on available API keys
2. Run queries against all configured models
3. Export results to an Excel file with timestamp

## Configuration

Models can be enabled/disabled in the `PRICING` configuration dictionary in `aisearchtest.py`.

## Output

Results are exported to Excel files with the following columns:
- Model
- Query
- Item (product name)
- Price (HUF)
- Source (website)
- Link (direct URL)
- Total USD cost (API usage)
- Elapsed seconds

## Development

Run tests:
```bash
pytest
```

## License

MIT
