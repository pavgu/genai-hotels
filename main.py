from langchain_community.document_loaders import JSONLoader
from bs4 import BeautifulSoup
import unicodedata

import os
from os import listdir, walk

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_google_vertexai import VertexAIEmbeddings

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

import json
from google.cloud import aiplatform

from langchain_chroma import Chroma

HOTEL_DIR = "hotel_information" # provide your source code path here

example_queries = [ "We are a group of 4 people who want to swim in a nice pool",
                   "We are a couple on our anniversary weekend",
                  ]

def parse_json(sources, schema, normalize=True):
    parsedList = []
    for item in sources:
        print(item)
        loader = JSONLoader(
            file_path=item,
            jq_schema=schema
        )
        parsedList += loader.load()
    if normalize:
        for item in parsedList:
            #item.page_content = BeautifulSoup(item.page_content, 'html.parser').getText(separator=' ')
            item.page_content = unicodedata.normalize("NFKD", BeautifulSoup(item.page_content, 'html.parser').getText(separator=','))
    return parsedList


def embed_text(model_name: str,
    text: str,
    task_type: str = "",
    title: str = "",
    output_dimensionality=None,
) -> list:
    """Generates a text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    text_embedding_input = TextEmbeddingInput(
        task_type=task_type, title=title, text=text
    )
    kwargs = (
        dict(output_dimensionality=output_dimensionality)
        if output_dimensionality
        else {}
    )
    embeddings = model.get_embeddings([text_embedding_input], **kwargs)
    return embeddings[0].values


def prepare_jsonl(sources, model, normalize=True):
    json_lines = ""

    for source in sources:
        with open(source) as data_file:
            data = json.load(data_file)

            for item in data['roomPrices']:
                id = item['id']

                desc = item['longEnglishDescription']

                if normalize:
                    desc = unicodedata.normalize("NFKD", BeautifulSoup(desc, 'html.parser').getText(separator=','))

                json_line = {}
                json_line['id'] = id
                json_line['embedding'] = embedded_text = embed_text(model_name=EMBEDDING_MODEL, text=desc, output_dimensionality=OUTPUT_DIMENSIONALITY)
                json_lines += json.dumps(json_line) + "\n"

    with open('output.json', 'w') as outfile:
        outfile.write(json_lines)


def run_similarity_search(query, database):
    docs = database.similarity_search(query)
    used_ids = set()
    for doc in docs:
        if doc.metadata.get('seq_num') not in used_ids:
            print(f'Metadata: {doc.metadata}; \nContent: {doc.page_content} \n')
            used_ids.add(doc.metadata.get('seq_num'))

def create_chroma_db(documents, embeddings, collection_name):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    db = Chroma(collection_name=collection_name)
    db.delete_collection()
    db = Chroma.from_documents(documents=documents, embedding=embeddings, collection_name=collection_name)
    return db

if __name__ == "__main__":
  ALL_JSON_FILES = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(HOTEL_DIR) for f in filenames]
  JSON_SOURCES = [s for s in ALL_JSON_FILES if s.endswith('/hotel_data.json')]
  # print(JSON_SOURCES)
  # Room description is given in the JSON arrays below
  roomPriceDescriptions = parse_json(JSON_SOURCES,'.roomPrices[].longEnglishDescription')
  print(len(roomPriceDescriptions))
  # Room addon description is given in the JSON arrays below
  roomAddonDescriptions = parse_json(JSON_SOURCES,'.roomAddons[].description')
  print(len(roomAddonDescriptions))

  # Example of using Vector Stores
  # MODEL_NAME="text-embedding-preview-0409"
  MODEL_NAME="textembedding-gecko@003"
  
  vertexAIembeddings = VertexAIEmbeddings(model_name=MODEL_NAME)

  EMBEDDING_MODEL = "text-embedding-preview-0409"
  #EMBEDDING_MODEL = "textembedding-gecko@003"
  TEXT = "Example text"
  OUTPUT_DIMENSIONALITY = 768
  
  # Get a text embedding for a downstream task.
  example_embedding = embed_text(model_name=EMBEDDING_MODEL,
      text=TEXT,
      output_dimensionality=OUTPUT_DIMENSIONALITY,
  )
  print(len(example_embedding))  # Expected value: {OUTPUT_DIMENSIONALITY}.

  prepare_jsonl(JSON_SOURCES, EMBEDDING_MODEL)
  
  INDEX_NAME = "vs-ba-index"
  
  # Before running the commands below, upload the JSONL files into the correct bucket
  my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
      display_name=INDEX_NAME,
      contents_delta_uri="gs://ba-vs-index",
      location="us-central1",
      dimensions=768,
      approximate_neighbors_count=10,
  )

  # Chroma VectorDB
  roomPriceDescriptionsDb = create_chroma_db(roomPriceDescriptions, vertexAIembeddings, 'Room_descriptions')
  roomAddonDescriptionsDb = create_chroma_db(roomAddonDescriptions, vertexAIembeddings, 'Addons_descriptions')

  print(len(roomPriceDescriptionsDb.get()['documents']))
  print(len(roomAddonDescriptionsDb.get()['documents']))

  query = "Kids"
  run_similarity_search(query, roomPriceDescriptionsDb)

  query = "Pets"
  run_similarity_search(query, roomAddonDescriptionsDb)
  
