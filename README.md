# songdesc
Python script that uses GPT to generate song descriptions

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. Prepare your input CSV file with the following columns:
   - `title` (required)
   - `artist` (required)
   - `album` (optional)
   - `year` (optional)

4. Run the script:
   ```bash
   python generate_song_descriptions_web.py
   ```

## Features

- Uses GPT-4o-mini for cost-effective, high-quality descriptions
- Web search enabled for accurate musical information
- Generates 300-500 character descriptions optimized for semantic search
- Includes objective musical features when available (tempo, key, instrumentation)
- Handles rate limiting and API errors gracefully

## Output

The script will create a new CSV file (`Songs_described.csv`) with an additional `semantic_description` column containing the generated descriptions.
