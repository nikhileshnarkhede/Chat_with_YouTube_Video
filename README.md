
# YouTube Transcript Q&A App

A **Streamlit-based web application** that allows users to interactively ask questions about any YouTube video using its transcript. The app leverages **LangChain**, **HuggingFace embeddings**, and **FAISS vector search** to provide accurate answers **directly from the video transcript**.

---

## Features

- Extracts transcript from YouTube videos automatically.
- Supports standard, short, and embedded YouTube URLs.
- Splits long transcripts into chunks for better retrieval.
- Embeds transcript chunks using **sentence-transformers/all-MiniLM-L6-v2**.
- Uses **FAISS** vector store with MMR (Maximal Marginal Relevance) search for diverse results.
- Interactive Q&A using **HuggingFace text-generation models**.
- Streamlit interface for easy web-based interaction.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/nikhileshnarkhede/Chat_with_YouTube_Video.git
cd Chat_with_YouTube_Video
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run Stremlit_Chat_with_YouTube_Video.py
```

2. Open the provided URL in your browser.  

3. Enter a **YouTube video URL**.  

4. Wait for the transcript to load, then **ask questions** about the video content.  

5. Answers are generated directly from the transcript.

---

## Supported URL Formats

- Standard YouTube: `https://www.youtube.com/watch?v=VIDEO_ID`
- Shortened URL: `https://youtu.be/VIDEO_ID`
- Embedded URL: `https://www.youtube.com/embed/VIDEO_ID`

---

## Project Structure

```
youtube-qa-app/
│
├── Stremlit_Chat_with_YouTube_Video.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── .gitignore            # Files to ignore in Git
└── README.md             # Project documentation
```

---

## Technologies Used

- [Streamlit](https://streamlit.io/) – Web application framework
- [LangChain](https://www.langchain.com/) – LLM orchestration
- [HuggingFace Transformers](https://huggingface.co/) – Embeddings & text generation
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/) – Transcript extraction
- Python 3.10+

---

## Notes

- Videos without transcripts or with disabled captions are not supported.
- For faster responses on repeated queries, consider caching embeddings.
- Ensure `.env` files (for API keys) are **not pushed to GitHub**.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Author

**Nikhilesh Narkhede** – [GitHub](https://github.com/nikhileshnarkhede) | [LinkedIn](https://www.linkedin.com/in/nikhileshnarkhede)
