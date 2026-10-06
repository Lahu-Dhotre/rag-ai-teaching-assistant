# RAG AI Teaching Assistant

A Python project that turns lecture videos into a searchable teaching assistant using transcript chunking, embeddings, and local LLM inference.

The workflow is simple: collect lecture videos, transcribe them, generate embeddings, retrieve the most relevant content for a question, and answer using a local generative model.

## Why this project exists

This project was built to make course material easier to explore and query. Instead of manually searching through long recordings, students or instructors can ask natural-language questions such as:

- What topics are covered in Video 3?
- Explain the section on React state management.
- Where is the concept of API integration discussed?

The system uses retrieval-augmented generation (RAG) to ground answers in the actual lecture transcript.

## Features

- Converts lecture videos to MP3 audio
- Uses OpenAI Whisper to transcribe spoken content
- Splits transcripts into time-stamped chunks
- Creates vector embeddings for each chunk
- Finds relevant content using cosine similarity
- Feeds the retrieved context into a local Ollama model
- Generates human-friendly answers grounded in the course material

## Tech stack

- Python
- FFmpeg
- OpenAI Whisper
- Ollama
- `bge-m3` embeddings model
- `llama3.2` generation model
- `pandas`, `numpy`, `scikit-learn`, `joblib`, `requests`

## Architecture overview

```text
videos/ --> video_to_mp3.py --> mp3 files
          \-> mp3_to_json.py --> transcript JSON chunks
                               \-> preprocess_json.py --> embeddings.joblib
                                                    \-> process_incoming.py
                                                           \-> local Ollama LLM
```

## Repository structure

```text
.
├── videos/                 # Raw lecture videos
├── jsons/                  # JSON transcript outputs from Whisper
├── embeddings.joblib       # Saved dataframe with chunk embeddings
├── video_to_mp3.py         # Converts videos to MP3 files
├── mp3_to_json.py          # Transcribes MP3 files to JSON chunks
├── preprocess_json.py      # Embeds transcript chunks for retrieval
├── process_incoming.py     # Accepts user questions and retrieves context
├── prompt.txt              # Prompt passed to the model
├── response.txt            # Final generated answer
├── .gitignore
├── Readme.md               # Project documentation
├── .venv/                  # Optional local virtual environment
└── requirements.txt       # Optional dependency file (if added later)
```

## Prerequisites

Before running the project, install:

- Python 3.9+
- FFmpeg
- Ollama
- Local Ollama models:
  - `bge-m3`
  - `llama3.2`

## Install dependencies

```bash
pip install pandas numpy scikit-learn joblib requests openai-whisper
```

If you use a virtual environment, activate it before running the scripts:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
pip install pandas numpy scikit-learn joblib requests openai-whisper
```

## Start Ollama

Make sure Ollama is installed and running locally:

```bash
ollama serve
```

Then pull the required models:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

## How the pipeline works

### 1. Add lecture videos

Place course videos inside the `videos/` folder.

### 2. Convert videos to MP3

```bash
python video_to_mp3.py
```

This script converts every video in `videos/` into audio files.

### 3. Transcribe MP3 files

```bash
python mp3_to_json.py
```

The transcription script uses Whisper to generate transcript segments with metadata such as:

- video number
- title
- start time
- end time
- transcript text

These are written into `jsons/` as JSON files.

### 4. Build the embedding index

```bash
python preprocess_json.py
```

This script:

- reads each JSON transcript file
- extracts the transcript chunks
- sends each chunk to the `bge-m3` embedding model
- stores the embeddings in `embeddings.joblib`

### 5. Ask a question

```bash
python process_incoming.py
```

The script will:

- accept a user question
- embed the query
- compare it with all stored chunk embeddings
- select top matching chunks using cosine similarity
- build a prompt with the relevant context
- ask the local LLM for a final answer

## Example usage

```bash
python process_incoming.py
```

Then enter a question like:

```text
What is discussed about JavaScript functions in the course?
```

The system returns a natural-language answer summarizing the relevant lesson sections.

## Notes about the current implementation

- The project currently assumes a local Ollama server at `http://localhost:11434`.
- The prompt is tuned for a web development or Sigma course context and can be customized for other subjects.
- The script writes `prompt.txt` and `response.txt` for debugging and inspection.
- The current retrieval logic focuses on top-k nearest chunks for a given user query.

## Potential improvements

- Add CLI arguments for model names, input folders, and output paths
- Support folder-based processing for multiple course modules
- Add a web UI or REST API
- Improve chunking and retrieval quality
- Add evaluation metrics for answer relevance
- Support multilingual content more explicitly

## License

This project is provided for educational and experimental use.

## Contributing

Contributions are welcome. If you improve the pipeline, add validation, or improve documentation, please open a pull request.
