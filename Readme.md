# RAG AI Teaching Assistant

A Python-based Retrieval-Augmented Generation (RAG) project for building a teaching assistant over lecture videos. The project extracts spoken content from video lessons, converts it into searchable chunks, creates vector embeddings, and answers student questions using a local LLM and semantic retrieval.

## Overview

This project is designed to help instructors or learners search and ask questions about course content from video lectures. It works by:

1. Collecting video files from the `videos/` directory
2. Converting videos to MP3 audio
3. Transcribing the audio into JSON chunks
4. Creating embeddings for each chunk
5. Using a vector similarity search to retrieve relevant content
6. Sending the retrieved context to a local LLM for a human-friendly answer

## Workflow

- `video_to_mp3.py` converts videos to MP3 files.
- `mp3_to_json.py` transcribes MP3 files using OpenAI Whisper and stores subtitle-like JSON chunks.
- `preprocess_json.py` creates embeddings for each text chunk and saves them in `embeddings.joblib`.
- `process_incoming.py` receives a question, finds the most relevant chunks using cosine similarity, and prompts a local LLM.

## Project structure

```text
.
├── jsons/                 # Transcribed JSON output for each audio/video
├── videos/                # Raw video files
├── embeddings.joblib      # Vectorized transcript chunks
├── video_to_mp3.py        # Video -> MP3 conversion
├── mp3_to_json.py         # MP3 -> transcript JSON
├── preprocess_json.py     # JSON -> embeddings
├── process_incoming.py    # Query the course material using RAG
├── prompt.txt             # Generated prompt for the LLM
├── response.txt           # Final model response
├── .gitignore
├── Readme.md              # Project documentation
```

## Requirements

- Python 3.9+
- FFmpeg installed and available in PATH
- Ollama installed and running locally
- Required Ollama models:
  - `bge-m3` for embeddings
  - `llama3.2` for generation
- Python dependencies:
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `requests`
  - `openai-whisper`

You can install the Python dependencies via:

```bash
pip install pandas scikit-learn joblib requests openai-whisper
```

## Setup

### 1. Add your videos

Place your lecture or lesson video files in the `videos/` folder.

### 2. Convert videos to MP3

```bash
python video_to_mp3.py
```

### 3. Transcribe audio into JSON

```bash
python mp3_to_json.py
```

### 4. Create embeddings

```bash
python preprocess_json.py
```

This creates the `embeddings.joblib` file used for semantic retrieval.

### 5. Ask questions

```bash
python process_incoming.py
```

When prompted, enter a question related to the lessons. The script will:

- embed the query using `bge-m3`
- compare it against stored embeddings
- fetch the most relevant transcript chunks
- generate a context-aware answer through the local LLM

## Example usage

```bash
python process_incoming.py
```

Then enter a prompt such as:

```text
What is covered in the React components lesson?
```

The assistant will answer by referencing the most relevant sections of the lecture transcript.

## Notes

- The project currently expects a local Ollama instance running on `http://localhost:11434`.
- The RAG prompt is tuned for a Sigma web development course, but it can be adapted for other subjects.
- The generated `prompt.txt` and `response.txt` files are useful for debugging and monitoring the LLM behavior.

## License

This project is provided as-is for educational and experimental use.

## Future improvements

- Add CLI arguments for model names and folders
- Support batch processing for large course libraries
- Add a web interface or REST API
- Improve transcript chunking and metadata handling
- Add evaluation metrics for retrieval quality
