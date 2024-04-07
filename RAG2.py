import streamlit as st
import fitz  # PyMuPDF
import spacy
import functools
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Load English NER model from spaCy
nlp = spacy.load("en_core_web_sm")

# Load RAG model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")
retriever = RagRetriever.from_pretrained("facebook/rag-token-base")
generator = RagTokenForGeneration.from_pretrained("facebook/rag-token-base")

# Cache decorator to memoize function results
def cache(func):
    memo = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in memo:
            memo[args] = func(*args)
        return memo[args]
    return wrapper

# Extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Preprocess text
def preprocess_text(text):
    # Remove newlines, extra whitespace, and other formatting artifacts
    cleaned_text = " ".join(text.strip().split())

    # Perform named entity recognition (NER) to identify entities in the text
    doc = nlp(cleaned_text)
    
    # Replace recognized entities with their entity types
    processed_text = ""
    for token in doc:
        if token.ent_type_:
            processed_text += token.ent_type_ + " "
        else:
            processed_text += token.text + " "
    
    return processed_text

# Retrieve relevant passages and cache the results
@cache
def retrieve_passages(query, context, n_docs):
    return retriever.retrieve(query, context=context, n_docs=n_docs)

# Generate answers
def generate_answers(query, retrieved_passages):
    input_ids = tokenizer.encode(query, retrieved_passages, return_tensors="pt")
    return generator.generate(input_ids)

# Display answers
def display_answers(generated_answers):
    for answer in generated_answers:
        return tokenizer.decode(answer, skip_special_tokens=True)

# Main function to process PDF and answer queries
def process_pdf_and_query(pdf_path, query):
    # Extract text from PDF
    pdf_text = extract_text_from_pdf(pdf_path)
    
    # Preprocess text
    cleaned_text = preprocess_text(pdf_text)
    context = retriever.encode_documents([cleaned_text])

    # Retrieve relevant passages
    retrieved_passages = retrieve_passages(query, context=context, n_docs=5)

    # Generate answers
    generated_answers = generate_answers(query, retrieved_passages)

    # Display answers
    return display_answers(generated_answers)

# Streamlit app
st.title("PDF Query Tool with RAG Model")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    st.write("PDF uploaded successfully!")
    
    # Query input
    query = st.text_input("Enter your query:")
    
    if st.button("Get Response"):
        # Process PDF and query
        response = process_pdf_and_query(uploaded_file, query)
        st.write("Response:", response)
