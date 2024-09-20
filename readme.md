# Pinecone Vectorstore Updater

This script updates a Pinecone vectorstore with new PDF documents from a Google Cloud Storage bucket. It keeps track of processed files to avoid duplicates and only processes new documents.

## Prerequisites

- Python 3.7+
- Google Cloud account with a storage bucket
- Pinecone account with an index created
- OpenAI API key

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/pinecone-vectorstore-updater.git
   cd pinecone-vectorstore-updater
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up your `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   CLOUD_STORAGE_BUCKET=your_google_cloud_storage_bucket_name
   ```

6. Authentication with Google Cloud:

   Option 1: Using `gcloud` CLI (recommended)
   - Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - Run `gcloud auth application-default login` and follow the prompts

   Option 2: Using a service account JSON key
   - Obtain a service account JSON key from the Google Cloud Console
   - Uncomment the following line in the script and update the path:
     ```python
     # credentials = service_account.Credentials.from_service_account_file('./path/to/your/credentials.json')
     ```
   - Update the `storage_client` initialization:
     ```python
     storage_client = storage.Client(credentials=credentials)
     ```

## Usage

Run the script with:

```bash
python import.py
```

The script will:
1. Connect to your Google Cloud Storage bucket
2. Check for new PDF files
3. Process new files and add them to your Pinecone vectorstore
4. Keep track of processed files to avoid duplicates in future runs

## How it works

1. The script loads a list of previously processed files from `processed_files.json`.
2. It compares this list with the files in your Google Cloud Storage bucket.
3. New files are downloaded, processed, and added to the Pinecone vectorstore.
4. The list of processed files is updated and saved for the next run.

## Customization

- Adjust the `CharacterTextSplitter` parameters in the `process_pdf` function to change how PDFs are split into chunks.
- Modify the `embeddings` initialization if you want to use a different embedding model.

## Troubleshooting

- If you encounter authentication issues, ensure you've set up Google Cloud authentication correctly.
- For Pinecone-related issues, check your API key and index name in the `.env` file.
- Make sure your Google Cloud Storage bucket contains PDF files.
- Make sure your model in pinecone matches your model in OpenAIEmbeddings()
- If you're having issues with package installations, ensure your virtual environment is activated and your `requirements.txt` file is up to date.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or encounter any problems. If someone knows what this chatter means, let me know lol 'Ignoring wrong pointing object 11 0 (offset 0)'. I think it is from pypdf but haven't dove in yet.

## License

[MIT License](https://opensource.org/licenses/MIT)