import os
import shutil
from pathlib import Path

import streamlit as st
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.embedders.ollama import (
    OllamaDocumentEmbedder,
    OllamaTextEmbedder,
)
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from ollama import generate

os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"


# this function is designed to create and return an instance of a ChromaDocumentStore
def get_doc_store():
    return ChromaDocumentStore(
        collection_name="mydocs",
        persist_path="./vec-index",
        distance_function="cosine",
    )


def get_context(query):
    """Retrieve and return context (relevant information) from a document store based on a given search query.

    It utilizes a pipeline that performs text embedding and document retrieval.
    """
    document_store = get_doc_store()

    query_pipeline = Pipeline()
    query_pipeline.add_component("text_embedder", OllamaTextEmbedder())
    query_pipeline.add_component("retriever", ChromaEmbeddingRetriever(document_store=document_store, top_k=3))

    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    result = query_pipeline.run({"text_embedder": {"text": query}})
    context = [doc.content for doc in result["retriever"]["documents"]]
    sources = [doc.meta["page_number"] for doc in result["retriever"]["documents"]]
    files = [doc.meta["file_path"] for doc in result["retriever"]["documents"]]
    final_context = [f"Context: {c} (Page: {s}, File: {f})" for c, s, f in zip(context, sources, files)]
    # Uncomment for debug st.write(final_context)
    return final_context


def indexing_pipe(filename):
    """Read a file, process it, and then index the resulting documents into a ChromaDocumentStore.

    It is part of document processing pipeline that consists of several steps,
    including converting the file to a document format, cleaning the content,
    splitting the document, embedding it, and finally writing it to the document store.
    """
    document_store = get_doc_store()

    pipeline = Pipeline()
    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component(
        "cleaner",
        DocumentCleaner(
            remove_empty_lines=True,
            remove_extra_whitespaces=True,
            remove_repeated_substrings=True,
        ),
    )
    pipeline.add_component(
        "splitter",
        DocumentSplitter(split_by="word", split_length=300, split_overlap=15),
    )
    pipeline.add_component("embedder", OllamaDocumentEmbedder())
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter", "cleaner")
    pipeline.connect("cleaner", "splitter")
    pipeline.connect("splitter", "embedder")
    pipeline.connect("embedder.documents", "writer")

    # Save the file to disk
    file_path = Path("uploads") / filename.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    pipeline.run({"converter": {"sources": [Path(file_path)]}})


def invoke_ollama(user_input):
    """Process user input and interact with a conversational AI model (e.g., a version of LLaMA).

    It also manages the chat interface and tracks the conversation state.
    """
    st.session_state.messages.append({"role": "user", "content": user_input})
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    system = f"""
        You are a helpful assistant that answers users questions and chats. 
        History has been provided in the <history> tags. 
        You must not mention your knowledge of the history to the user,
        only use it to answer follow up questions if needed.
        {{history}}
        {st.session_state.messages}
        {{history}}

        Context to help you answer user's questions have been provided in the <context> tags.
        {{context}}
        {get_context(user_input)}
        {{context}}
        Use ONLY the history and or context provided to answer the question.
        Use as few words as possible to accurately answer.
        """
    # Uncomment to make llama use a template {"answer": "the answer"}
    # Use the following template: {json.dumps(template)}."""
    prompt_wrapper = f"""
        You are a helpful assistant that answers users questions and chats. 
        Use the provided history and context to answer the question. 
        {{user_query}}
        {user_input}
        {{user_query}}
        Use as few words as possible to accurately answer, providing citations to the page number 
        and file path from which your answer was synthesized.
        """

    data = {
        "prompt": prompt_wrapper,
        "model": "llama3.2:3b",
        "stream": True,
        "options": {"top_p": 0.05, "top_k": 5},
    }
    s = ""
    box = st.chat_message("assistant").empty()

    for part in generate(
        model=data["model"],
        prompt=data["prompt"],
        system=system,
        # Format seems equivelant to enforcing a template within the prompt
        # format=data["format"],
        options=data["options"],
        stream=data["stream"],
    ):
        s += part["response"]
        box.write(s)

    st.session_state.messages.append({"role": "assistant", "content": s})


# clear_convo function is designed to clear the conversation history
def clear_convo():
    st.session_state["messages"].clear()


def clear_uploads_and_chroma():
    """Clear uploads folder and Chroma collection content (keeps vec-index dir)."""
    uploads = Path("uploads")
    if not uploads.exists() or not any(uploads.iterdir()):
        return
    doc_store = get_doc_store()
    docs = doc_store.filter_documents(
        filters={"field": "source", "operator": "==", "value": "upload"},
    )
    if docs:
        ids = [doc.id for doc in docs]
        doc_store.delete_documents(ids=ids)

    if uploads.exists():
        shutil.rmtree(uploads)
    st.session_state["clear_toast_message"] = True
    st.session_state["needs_rerun"] = True


# init function is designed to initialize the Streamlit app and set the page configuration
def init():
    st.set_page_config(page_title="Local Llama", page_icon=":robot_face: ")
    st.sidebar.title("Local Llama")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    # Initialize session state for file list
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    Path("uploads").mkdir(exist_ok=True, parents=True)


def get_uploaded_files():
    if uploaded_files := sorted(Path("uploads").iterdir(), key=lambda x: x.name.lower()):
        uploaded_files = "<br>".join(f"**{i}.** {f.name}" for i, f in enumerate(uploaded_files, start=1))
    else:
        uploaded_files = "No files uploaded."
    st.session_state.uploaded_files = uploaded_files
    return len(list(Path("uploads").iterdir()))


# Function to update expander content dynamically on uploaded file change
def update_expander():
    n = get_uploaded_files()
    with expander_placeholder.container():
        with st.expander(f"Uploaded Files ({n})", expanded=False):
            uploaded_files = st.empty()
            uploaded_files.markdown(st.session_state.uploaded_files, unsafe_allow_html=True)


# main function
if __name__ == "__main__":
    init()
    if st.session_state.get("needs_rerun"):
        del st.session_state["needs_rerun"]
        st.rerun()
    # Show upload success toast from previous run (so it appears after rerun that clears file input)
    if msg := st.session_state.pop("upload_toast_message", None):
        if hasattr(st, "toast"):
            st.toast(f"Indexed {msg}!", duration=5)
        else:
            st.success(f"Indexed {msg}!")
    # Show toast when Clear Indexed Files was pressed
    if st.session_state.pop("clear_toast_message", None):
        if hasattr(st, "toast"):
            st.toast("Uploads and index cleared.", duration=5)
        else:
            st.success("Uploads and index cleared.")
    file = st.file_uploader(
        "Choose a file to index...",
        type=["docx", "pdf", "txt", "md"],
        key=f"file_upload_{st.session_state['file_uploader_key']}",
    )

    # display on sidebar all files within uploads dir
    uploaded_files = None
    expander_placeholder = None

    _page_help = (
        "This app stores uploaded files in the **uploads** folder and indexes them "
        "into a local Chroma document store so you can reuse your documents for Q&A."
    )
    with st.sidebar:
        clear_button = st.button("Clear Conversation", key="clear", on_click=clear_convo)
        st.button(
            "Clear Indexed Files",
            key="clear_data",
            on_click=clear_uploads_and_chroma,
            type="secondary",
        )
        # Create a placeholder for the expander
        expander_placeholder = st.empty()
        update_expander()
        st.caption(_page_help)

    clicked = st.button("Upload File", key="Upload")
    if file and clicked:
        with st.spinner("Please wait..."):
            indexing_pipe(file)
        st.session_state["file_uploader_key"] += 1  # clear file uploader
        st.session_state["upload_toast_message"] = file.name
        update_expander()
        st.rerun()

    user_input = st.chat_input("Say something")

    if user_input:
        invoke_ollama(user_input=user_input)
