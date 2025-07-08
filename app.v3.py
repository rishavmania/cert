import streamlit as st
import pandas as pd
import os
import json
import openai
from openai import OpenAI
import io
import psycopg2 # For PostgreSQL connection
from pgvector.psycopg2 import register_vector # For pgvector support
from pyairtable import Table # For Airtable

# --- Configuration Constants ---
# Directory where processed JSON data is stored (still needed for metadata JSONs)
PROCESSED_DATA_DIR = r"C:\Users\drris\Downloads\SLapp\pdfs\processed_data"

# OpenAI API Key for LLM calls (gpt-3.5-turbo) and Query Embeddings
# IMPORTANT: Replace with your actual OpenAI API Key
OPENAI_API_KEY_LLM = "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"
openai_client_llm = OpenAI(api_key=OPENAI_API_KEY_LLM)

# Supabase Configuration
# IMPORTANT: Replace with your Supabase connection details (use st.secrets for cloud deployment)
SUPABASE_DB_HOST = "db.ckspzmwuwqcyygkpatwb.supabase.co" # e.g., "db.abcdefghijklm.supabase.co"
SUPABASE_DB_PORT = "5432"
SUPABASE_DB_NAME = "postgres"
SUPABASE_DB_USER = "postgres"
SUPABASE_DB_PASSWORD = "0HKAulx7QO7vmMje"
SUPABASE_TABLE_NAME = "documents" # The table where your chunks are stored

# Airtable Configuration
# IMPORTANT: Replace with your actual Airtable API Key, Base ID, and Table Name
AIRTABLE_API_KEY = "patBH8wgnEl2oMAEr.59c81f22f57ce701f65ac52063bf3118b2d1ffe6540c1a6e5c52e2ec9cfc1bb6" # e.g., "patXXXXXXXXXXXXXXX"
AIRTABLE_BASE_ID = "appLMjllbt9R613Xd" # e.g., "appXXXXXXXXXXXXXXX"
AIRTABLE_REQUIREMENTS_TABLE_NAME = "Requirements" # e.g., "Requirements" or "Checklist"
AIRTABLE_REQUIREMENT_TEXT_COLUMN = "Requirement Statement" # Column name in Airtable for the requirement text
AIRTABLE_KEYWORDS_COLUMN = "Keywords" # Column name in Airtable for the keywords
AIRTABLE_RELATED_DOCS_COLUMN = "Document" # Column name in Airtable for related documents (e.g., "TC19-16")


# RAG Configuration
TOP_K_CHUNKS = 10 # Number of top relevant chunks to retrieve from Supabase
MAX_CHAT_HISTORY_MESSAGES = 5 # Max number of previous user/assistant turns to send to LLM


# --- Supabase Database Connection ---
@st.cache_resource # Cache the Supabase connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=SUPABASE_DB_HOST,
            port=SUPABASE_DB_PORT,
            database=SUPABASE_DB_NAME,
            user=SUPABASE_DB_USER,
            password=SUPABASE_DB_PASSWORD
        )
        register_vector(conn) # Enable pgvector support
        print("Successfully connected to Supabase PostgreSQL for app.")
        return conn
    except Exception as e:
        st.error(f"Error connecting to Supabase PostgreSQL: {e}. Please check your Supabase connection details and ensure pgvector extension is enabled.")
        st.stop()

supabase_conn = get_db_connection()


# --- Initialize Airtable Client and Fetch Data ---
@st.cache_data(ttl=3600) # Cache Airtable data for 1 hour to reduce API calls
def get_airtable_requirements_data(api_key, base_id, table_name, req_text_col, keywords_col, related_docs_col):
    if not api_key or api_key == "YOUR_AIRTABLE_API_KEY_HERE":
        st.warning("Airtable API key is not set. Airtable integration will be skipped.")
        return None, {}

    try:
        table = Table(api_key, base_id, table_name)
        all_records = table.all()
        print(f"Airtable table '{table_name}' fetched successfully. Found {len(all_records)} records.")

        description_to_info_map = {}
        for record in all_records:
            fields = record.get('fields', {})
            
            req_text = fields.get(req_text_col, "")
            keywords = fields.get(keywords_col, "")
            related_docs = fields.get(related_docs_col, "")

            if isinstance(req_text, list):
                req_text = ", ".join(map(str, req_text))
            req_text = str(req_text).strip()

            if isinstance(keywords, list):
                keywords = ", ".join(map(str, keywords))
            keywords = str(keywords).strip()

            if isinstance(related_docs, list):
                if related_docs and isinstance(related_docs[0], dict):
                    related_docs = ", ".join([d.get('name', d.get('id', '')) for d in related_docs])
                else:
                    related_docs = ", ".join(map(str, related_docs))
            related_docs = str(related_docs).strip()
            
            if req_text:
                description_to_info_map[req_text] = {
                    "keywords": keywords,
                    "related_docs": related_docs
                }
        return all_records, description_to_info_map
    except Exception as e:
        st.error(f"Error initializing or fetching Airtable data from '{table_name}': {e}. Please check your Airtable API Key, Base ID, Table Name, and column names.")
        return None, {}

airtable_all_records, airtable_description_to_info_map = get_airtable_requirements_data(
    AIRTABLE_API_KEY,
    AIRTABLE_BASE_ID,
    AIRTABLE_REQUIREMENTS_TABLE_NAME,
    AIRTABLE_REQUIREMENT_TEXT_COLUMN,
    AIRTABLE_KEYWORDS_COLUMN,
    AIRTABLE_RELATED_DOCS_COLUMN
)


# --- Helper function to load data from JSON files (for metadata, not chunks) ---
def load_data_from_json(filename):
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Processed data file not found: {file_path}. Please run process_vectorize_pdfs.py first.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. File might be corrupted or empty.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading {file_path}: {e}")
        st.stop()

# --- Load all processed metadata (excluding chunks, which are in Supabase) ---
documents_metadata = load_data_from_json("processed_documents_metadata.json")
checklist_data = load_data_from_json("checklist_items.json")
discrepancy_data = load_data_from_json("discrepancies.json")
extracted_entities_data = load_data_from_json("extracted_entities.json")


# Helper to map doc IDs to titles from loaded metadata
def get_doc_title(doc_id):
    for doc in documents_metadata:
        if doc['id'] == doc_id:
            return doc['title']
    return "Unknown Document"

# Helper to get full content from loaded metadata (for display, not RAG)
def get_full_doc_content(doc_id):
    for doc in documents_metadata:
        if doc['id'] == doc_id:
            return doc.get('full_text_content', 'Content not available.')
    return "Document content not found."

# Helper to get original PDF file path for local links
def get_original_pdf_file_path(doc_id):
    for doc in documents_metadata:
        if doc['id'] == doc_id:
            return doc.get('original_pdf_file_path', '')
    return ""

# Helper to get extracted tables for a document
def get_extracted_tables(doc_id):
    for doc in documents_metadata:
        if doc['id'] == doc_id:
            return doc.get('extracted_tables', [])
    return []

# Helper to get extracted image paths for a document
def get_extracted_image_paths(doc_id):
    for doc in documents_metadata:
        if doc['id'] == doc_id:
            return doc.get('extracted_image_paths', [])
    return []

# --- OpenAI Embedding Function for Query ---
def get_query_embedding(text):
    """Generates an embedding for the given query text using OpenAI's API."""
    if not OPENAI_API_KEY_LLM or OPENAI_API_KEY_LLM == "YOUR_RANDOM_OPENAI_API_KEY_HERE":
        st.error("OpenAI API key not set for query embeddings.")
        return None
    try:
        response = openai_client_llm.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return None

# --- RAG Q&A Function ---
def ask_rag_with_supabase(query, chat_history):
    """
    Performs RAG: retrieves relevant chunks from Supabase and Airtable,
    and uses them as context for OpenAI LLM.
    """
    if not openai_client_llm.api_key or openai_client_llm.api_key == "YOUR_RANDOM_OPENAI_API_KEY_HERE":
        return "Please set your OpenAI API key in the app.py file to use this feature.", []

    context_parts = []
    source_info = set() # To keep track of unique sources (Doc IDs, Airtable records)

    # 1. Retrieve relevant chunks from Supabase (PDFs)
    retrieved_chunks_data = []
    try:
        query_embedding = get_query_embedding(query)
        if query_embedding is None:
            context_parts.append("\n\n--- Failed to generate embedding for PDF retrieval. ---\n\n")
        else:
            cur = supabase_conn.cursor()
            # Use the <-> operator for L2 distance (Euclidean distance)
            # ORDER BY embedding <-> %s LIMIT %s finds the closest vectors
            cur.execute(
                f"SELECT doc_id, doc_title, chunk_index, content FROM {SUPABASE_TABLE_NAME} ORDER BY embedding <-> %s LIMIT %s;",
                (query_embedding, TOP_K_CHUNKS)
            )
            retrieved_chunks_data = cur.fetchall()
            cur.close()

            if retrieved_chunks_data:
                context_parts.append("\n\n--- Retrieved Document Context (from PDFs) ---\n\n")
                for doc_id, doc_title, chunk_index, content in retrieved_chunks_data:
                    source_info.add(f"PDF: {doc_title}")
                    context_parts.append(f"Document: {doc_title} (Chunk {chunk_index})\nContent:\n{content}\n")
            else:
                context_parts.append("\n\n--- No relevant PDF documents found in the knowledge base. ---\n\n")

    except Exception as e:
        print(f"Error during Supabase retrieval: {e}")
        context_parts.append(f"\n\n--- Error retrieving from PDF knowledge base (Supabase): {e} ---\n\n")

    # 2. Retrieve relevant records from Airtable (if configured)
    relevant_airtable_records = [] # Initialize here for consistent printing in debug
    if airtable_all_records: # Check if Airtable data was successfully loaded
        try:
            query_lower = query.lower()
            for record in airtable_all_records:
                fields = record.get('fields', {})
                req_text = str(fields.get(AIRTABLE_REQUIREMENT_TEXT_COLUMN, "")).lower()
                keywords = str(fields.get(AIRTABLE_KEYWORDS_COLUMN, "")).lower()
                
                # Simple keyword matching for demonstration
                if query_lower in req_text or any(word.strip() in query_lower for word in keywords.split(',')):
                    relevant_airtable_records.append(record)

            if relevant_airtable_records:
                context_parts.append("\n\n--- Retrieved Structured Context (from Airtable) ---\n\n")
                for record in relevant_airtable_records:
                    fields = record.get('fields', {})
                    req_text = str(fields.get(AIRTABLE_REQUIREMENT_TEXT_COLUMN, "N/A")).strip()
                    record_id = record.get('id', 'N/A')
                    related_docs_from_airtable = str(fields.get(AIRTABLE_RELATED_DOCS_COLUMN, "N/A")).strip()
                    
                    source_info.add(f"Airtable Requirement ID: {record_id} (Related Doc: {related_docs_from_airtable})")

                    context_parts.append(f"Airtable Requirement (ID: {record_id}): {req_text}\n")
            else:
                context_parts.append("\n\n--- No relevant Airtable requirements found. ---\n\n")
        except Exception as e:
            print(f"Error during Airtable retrieval: {e}")
            context_parts.append(f"\n\n--- Error retrieving from Airtable: {e} ---\n\n")
    else:
        context_parts.append("\n\n--- Airtable integration not configured or failed to initialize. ---\n\n")


    final_context = "".join(context_parts)

    # 3. Prepare messages for OpenAI LLM
    messages = [
        {"role": "system", "content": "You are an expert assistant for hydrogen aircraft certification. Your goal is to provide comprehensive and synthesized answers based on the provided document context and structured data. Combine information from all relevant sources to answer the user's question. If the information is not explicitly available in the provided context, state that you don't have enough information from the given sources. Do not make up information."}
    ]
    
    for msg in chat_history[-MAX_CHAT_HISTORY_MESSAGES:]:
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": f"Based on the following information, answer this question: {query}\n\n{final_context}"})

    # --- Debugging prints ---
    print(f"--- OpenAI API Call Debug (RAG) ---")
    print(f"Query: {query}")
    print(f"Number of retrieved PDF chunks: {len(retrieved_chunks_data)}")
    print(f"Number of retrieved Airtable records: {len(relevant_airtable_records)}")
    print(f"Length of combined context text: {len(final_context)} characters")
    print(f"Total messages sent to LLM: {len(messages)}")
    print(f"------------------------------------")

    # 4. Call OpenAI LLM
    try:
        response = openai_client_llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1500,
            temperature=0.7
        )
        answer = response.choices[0].message.content 
        return answer, list(source_info)
    except openai.APIStatusError as e:
        print(f"OpenAI APIStatusError: Status {e.status_code}, Response: {e.response}")
        return f"OpenAI API Error (Status {e.status_code}): {e.response.json().get('error', {}).get('message', 'Unknown error details')}", []
    except openai.AuthenticationError:
        print("OpenAI AuthenticationError: Invalid OpenAI API key.")
        return "Authentication Error: Invalid OpenAI API key. Please check your key in app.py.", []
    except openai.RateLimitError:
        print("OpenAI RateLimitError: Rate limit exceeded.")
        return "Rate Limit Exceeded: You've sent too many requests. Please wait and try again.", []
    except Exception as e:
        print(f"An unexpected error occurred with the OpenAI API: {e}")
        return f"An unexpected error occurred with the OpenAI API: {e}", []


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Hydrogen Certification Dashboard")

st.title("Hydrogen Aircraft Certification Assistant Dashboard")

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Checklist Overview", "Discrepancy Tracker", "Document Explorer", "Document Comparator", "Ask a Question", "Generate Clarity Request"])

# Global Filters (can be moved to main content if preferred)
st.sidebar.header("Global Filters")
st.sidebar.selectbox("Aircraft System", ["All", "Hydrogen Fuel Cell", "PEM Fuel Cell System", "Wing Structure"])
st.sidebar.selectbox("FAR Part", ["All", "Part 25", "Part 23", "Part 91"])
st.sidebar.selectbox("Severity", ["All", "Critical", "Major", "Minor"])

# --- Main Content Area ---
if page == "Ask a Question":
    st.header("Ask a Question (RAG Chatbot)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.info("The AI will automatically retrieve relevant information from your documents (via Supabase) and Airtable to answer your questions.")

    if prompt := st.chat_input("Ask a question about the documents and requirements..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Retrieving information and asking AI..."):
            answer, sources_used = ask_rag_with_supabase(prompt, st.session_state.messages) # Changed function call
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources_used:
                    st.caption(f"Answer based on: {', '.join(sources_used)}")
                else:
                    st.caption("No specific sources found relevant to this query.")


elif page == "Checklist Overview":
    st.header("Certification Requirements Checklist")
    df_checklist = pd.DataFrame(checklist_data)

    st.subheader("Filters:")
    search_term = st.text_input("Search checklist items:")
    status_filter = st.selectbox("Filter by Status:", ["All", "Pending", "In Progress", "Completed", "Requires Clarity"])

    filtered_df = df_checklist.copy()
    if search_term:
        filtered_df = filtered_df[filtered_df['description'].str.contains(search_term, case=False)]
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]

    # Add Keywords column by mapping from Airtable data
    filtered_df['Keywords'] = filtered_df['description'].apply(
        lambda desc: airtable_description_to_info_map.get(desc.strip(), {}).get(AIRTABLE_KEYWORDS_COLUMN, "N/A")
    )
    # Also get related docs from Airtable for better accuracy if available
    filtered_df['Related Documents (from Airtable)'] = filtered_df['description'].apply(
        lambda desc: airtable_description_to_info_map.get(desc.strip(), {}).get(AIRTABLE_RELATED_DOCS_COLUMN, "N/A")
    )

    filtered_df['Related Entities'] = filtered_df['related_entities'].apply(lambda x: ", ".join(x))

    display_cols = ['id', 'description', 'Keywords', 'Related Documents (from Airtable)', 'relevant_section', 'Related Entities', 'status', 'associated_discrepancy', 'notes']
    st.dataframe(filtered_df[display_cols], height=400, use_container_width=True)

    st.subheader("Update Checklist Item (Example)")
    item_to_update = st.selectbox("Select Item ID to update:", df_checklist['id'].tolist())
    new_status = st.selectbox("New Status:", ["Pending", "In Progress", "Completed", "Requires Clarity"])
    if st.button(f"Update {item_to_update}"):
        st.info(f"Feature to update status for '{item_to_update}' to '{new_status}' would be here (not saved in this prototype).")


elif page == "Discrepancy Tracker":
    st.header("Identified Discrepancies & Clarity Requests")
    df_discrepancies = pd.DataFrame(discrepancy_data)

    st.subheader("Filters:")
    disc_search_term = st.text_input("Search discrepancies:")
    disc_status_filter = st.selectbox("Filter by Clarity Status:", ["All", "Not Sent", "Drafted", "Sent to FAA", "Response Received", "Resolved"])

    filtered_discs = df_discrepancies.copy()
    if disc_search_term:
        filtered_discs = filtered_discs[filtered_discs['nature_of_conflict'].str.contains(disc_search_term, case=False)]
    if disc_status_filter != "All":
        filtered_discs = filtered_discs[filtered_discs['clarity_request_status'] == disc_status_filter]

    filtered_discs['Involved Documents'] = filtered_discs['involved_docs'].apply(lambda doc_ids: ", ".join([get_doc_title(d_id) for d_id in doc_ids]))
    filtered_discs['Conflicting Requirements'] = filtered_discs['conflicting_requirements'].apply(lambda req_ids: ", ".join(req_ids))

    display_cols_disc = ['id', 'Conflicting Requirements', 'nature_of_conflict', 'Involved Documents', 'proposed_resolution_notes', 'clarity_request_status']
    st.dataframe(filtered_discs[display_cols_disc], height=400, use_container_width=True)

    st.subheader("Generate Clarity Request (Example)")
    selected_disc = st.selectbox("Select Discrepancy ID for request:", df_discrepancies['id'].tolist())
    if st.button(f"Generate Draft for {selected_disc}"):
        disc_details = df_discrepancies[df_discrepancies['id'] == selected_disc].iloc[0]
        st.info(f"Drafting clarity request for Discrepancy ID: {selected_disc}\n"
                f"Nature: {disc_details['nature_of_conflict']}\n"
                f"This would generate a document (PDF/Word) with details for FAA.")


elif page == "Document Explorer":
    st.header("Document Library & Extracted Insights")
    df_docs = pd.DataFrame(documents_metadata)

    st.subheader("Search Documents:")
    doc_search = st.text_input("Search document titles or topics:")
    filtered_docs = df_docs.copy()
    if doc_search:
        filtered_docs = filtered_docs[
            filtered_docs['title'].str.contains(doc_search, case=False) |
            filtered_docs['key_topics'].apply(lambda x: any(doc_search.lower() in topic.lower() for topic in x))
        ]

    for index, row in filtered_docs.iterrows():
        original_pdf_path = get_original_pdf_file_path(row['id'])
        if original_pdf_path and os.path.exists(original_pdf_path):
            st.markdown(f"**[{row['title']}](file://{os.path.abspath(original_pdf_path)})** (Click to open original PDF - browser security may restrict)")
        else:
            st.markdown(f"**{row['title']}** (Original PDF not found at: {original_pdf_path})")

        st.write(f"**Key Topics:** {', '.join(row['key_topics'])}")
        st.write(f"**Requirements Identified:** {row['num_requirements']}")
        st.write(f"**Discrepancies Detected:** {row['num_discrepancies']}")
        
        content_preview = row.get('full_text_content', 'Content not available.')
        st.write(f"**Sample Text Content:** *{content_preview[:300]}...*")

        tables = get_extracted_tables(row['id'])
        if tables:
            st.subheader(f"Extracted Tables ({len(tables)}):")
            for i, table in enumerate(tables):
                st.markdown(f"**Table {i+1}:**")
                try:
                    df_table = pd.DataFrame(table['data'])
                    st.dataframe(df_table, use_container_width=True)
                except Exception as e:
                    st.write(f"Could not display table data: {e}")
                
        images = get_extracted_image_paths(row['id'])
        if images:
            st.subheader(f"Extracted Images ({len(images)}):")
            for img_path in images:
                if os.path.exists(img_path):
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
                else:
                    st.write(f"Image not found: {os.path.basename(img_path)}")
        
        st.markdown("---")

elif page == "Document Comparator":
    st.header("Compare Two Documents")
    
    doc_options = {doc['title']: doc['id'] for doc in documents_metadata}
    
    if len(doc_options) < 2:
        st.warning("Please process at least two documents to use the Document Comparator.")
    else:
        selected_title_1 = st.selectbox("Select Document 1:", list(doc_options.keys()), index=0)
        default_idx_2 = 1 if len(doc_options) > 1 else 0
        if selected_title_1 == list(doc_options.keys())[default_idx_2] and len(doc_options) > 2:
            default_idx_2 = 2
        selected_title_2 = st.selectbox("Select Document 2:", list(doc_options.keys()), index=default_idx_2)


        doc_id_1 = doc_options[selected_title_1]
        doc_id_2 = doc_options[selected_title_2]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Document 1: {selected_title_1}")
            st.text_area("Full Text Content:", get_full_doc_content(doc_id_1), height=300)
            st.markdown(f"**Key Topics:** {', '.join([d['key_topics'] for d in documents_metadata if d['id'] == doc_id_1][0])}")
            st.markdown(f"**Extracted Entities (Simulated):**")
            for entity in extracted_entities_data.get(doc_id_1, []):
                st.markdown(f"- {entity['text']} (`{entity['type']}`)")
            
            st.markdown("**Extracted Tables:**")
            tables1 = get_extracted_tables(doc_id_1)
            if tables1:
                for i, table in enumerate(tables1):
                    st.markdown(f"Table {i+1}:")
                    try:
                        st.dataframe(pd.DataFrame(table['data']), use_container_width=True)
                    except Exception as e:
                        st.write(f"Could not display table data: {e}")
            else:
                st.write("No tables extracted.")

            st.markdown("**Extracted Images:**")
            images1 = get_extracted_image_paths(doc_id_1)
            if images1:
                for img_path in images1:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.write("No images extracted.")


        with col2:
            st.subheader(f"Document 2: {selected_title_2}")
            st.text_area("Full Text Content:", get_full_doc_content(doc_id_2), height=300)
            st.markdown(f"**Key Topics:** {', '.join([d['key_topics'] for d in documents_metadata if d['id'] == doc_id_2][0])}")
            st.markdown(f"**Extracted Entities (Simulated):**")
            for entity in extracted_entities_data.get(doc_id_2, []):
                st.markdown(f"- {entity['text']} (`{entity['type']}`)")

            st.markdown("**Extracted Tables:**")
            tables2 = get_extracted_tables(doc_id_2)
            if tables2:
                for i, table in enumerate(tables2):
                    st.markdown(f"Table {i+1}:")
                    try:
                        st.dataframe(pd.DataFrame(table['data']), use_container_width=True)
                    except Exception as e:
                        st.write(f"Could not display table data: {e}")
            else:
                st.write("No tables extracted.")

            st.markdown("**Extracted Images:**")
            images2 = get_extracted_image_paths(doc_id_2)
            if images2:
                for img_path in images2:
                    if os.path.exists(img_path):
                        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.write("No images extracted.")

        st.markdown("---")
        st.subheader("Comparison Result (Simulated Discrepancy Detection)")

        # Updated comparison logic to use the exact titles loaded from documents_metadata
        if ("1. tc19-16 - Energy Supply Device ARC Recommendation Report.pdf" in selected_title_1 and "2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.pdf" in selected_title_2) or \
           ("2. tc18-49 - Failure Mode and Effects Analysis on PEM Fuel Cell Systems for Aircraft Power Applications.pdf" in selected_title_1 and "1. tc19-16 - Energy Supply Device ARC Recommendation Report.pdf" in selected_title_2):
            st.warning("🚨 **Potential Discrepancy Detected!**")
            st.markdown("**Nature of Conflict:**")
            st.write("These documents likely present a temperature operating range discrepancy. One may allow up to +85°C for hydrogen fuel cells, while the other specifies a maximum of 80°C for PEM fuel cell systems. This requires clarity for PEM-specific limits.")
            st.write("This aligns with **Discrepancy ID: `disc_001`** in the Discrepancy Tracker.")
        elif ("tc19-16.pdf" in selected_title_1 and "tc18-49.pdf" in selected_title_2) or \
             ("tc18-49.pdf" in selected_title_1 and "tc19-16.pdf" in selected_title_2):
            st.info("These are general documents (based on their short titles). Your AI would analyze their content for subtle relationships or potential overlaps. No direct conflict defined in fake data.")
        elif ("tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.pdf" in selected_title_1 and "tc16-24.pdf" in selected_title_2) or \
             ("tc16-24.pdf" in selected_title_1 and "tc16-24 - Abusive Testing of Proton Exchange Membrane Hydrogen Fuel Cells.pdf" in selected_title_2):
            st.info("These documents share a common number (tc16-24). Your AI could confirm if the general 'tc16-24.txt' document acts as a summary or a foundational document for the more specific 'Abusive Testing' report.")
        elif ("tc21-3 - Fuel Tank Flammability Assessment Method User's Manual - Updated for Version 11.pdf" in selected_title_1 and "tc20-9 - Aircraft Fuel Cell and Safety Management System.pdf" in selected_title_2) or \
             ("tc20-9 - Aircraft Fuel Cell and Safety Management System.pdf" in selected_title_1 and "tc21-3 - Fuel Tank Flammability Assessment Method User's Manual - Updated for Version 11.pdf" in selected_title_2):
            st.info("These documents are related to fuel tank flammability and overall aircraft fuel cell safety management. Your AI could identify how specific flammability assessments feed into broader safety protocols.")
        else:
            st.info("No direct *pre-defined* conflict in fake data for these specific documents. Your AI would perform deeper analysis to find subtle relationships or inconsistencies.")


elif page == "Generate Clarity Request":
    st.header("Generate New Clarity Request")
    st.write("This section would allow you to draft a new clarity request from scratch.")
    request_title = st.text_input("Request Title:")
    request_details = st.text_area("Detailed Request:")
    st.selectbox("Relates to Discrepancy ID (Optional):", ["None"] + [d['id'] for d in discrepancy_data])
    st.selectbox("Relates to Checklist Item ID (Optional):", ["None"] + [r['id'] for r in checklist_data])
    if st.button("Generate Draft Document"):
        st.info(f"Draft for '{request_title}' generated. (This would create a PDF/Word file).")

