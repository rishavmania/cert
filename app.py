import streamlit as st
import pandas as pd
import os
import json
import openai # Import OpenAI library
from openai import OpenAI # Import the new OpenAI client
import io # For displaying images from bytes

# --- OpenAI API Configuration ---
# IMPORTANT: Replace "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" with your actual OpenAI API Key
# Initialize the OpenAI client
client = OpenAI(api_key="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")

# --- Configuration for processed data ---
PROCESSED_DATA_DIR = r"C:\Users\drris\Downloads\SLapp\pdfs\processed_data"
PDF_DIR = r"C:\Users\drris\Downloads\SLapp\pdfs" # Used for constructing image paths

# --- Helper function to load data from JSON files ---
def load_data_from_json(filename):
    file_path = os.path.join(PROCESSED_DATA_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: Processed data file not found: {file_path}. Please run process_pdfs.py first.")
        st.stop() # Stop execution if critical files are missing
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. File might be corrupted or empty.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred loading {file_path}: {e}")
        st.stop()

# --- Load all processed data ---
documents_data = load_data_from_json("processed_documents_metadata.json")
checklist_data = load_data_from_json("checklist_items.json")
discrepancy_data = load_data_from_json("discrepancies.json")
extracted_entities_data = load_data_from_json("extracted_entities.json") # Load extracted entities


# Helper to map doc IDs to titles from loaded data
def get_doc_title(doc_id):
    for doc in documents_data:
        if doc['id'] == doc_id:
            return doc['title']
    return "Unknown Document"

# Helper to get full content from loaded data
def get_full_doc_content(doc_id):
    for doc in documents_data:
        if doc['id'] == doc_id:
            return doc.get('full_text_content', 'Content not available.')
    return "Document content not found."

# Helper to get original PDF file path for local links
def get_original_pdf_file_path(doc_id):
    for doc in documents_data:
        if doc['id'] == doc_id:
            return doc.get('original_pdf_file_path', '')
    return ""

# Helper to get extracted tables for a document
def get_extracted_tables(doc_id):
    for doc in documents_data:
        if doc['id'] == doc_id:
            return doc.get('extracted_tables', [])
    return []

# Helper to get extracted image paths for a document
def get_extracted_image_paths(doc_id):
    for doc in documents_data:
        if doc['id'] == doc_id:
            return doc.get('extracted_image_paths', [])
    return []

# --- OpenAI Q&A Function (RAG Concept) ---
def ask_openai_rag(query, documents_context_ids, chat_history):
    """
    Asks OpenAI a question, providing document content as context and chat history.
    Uses OpenAI API v1.0.0+ syntax.
    """
    # Check if API key is set
    if not client.api_key or client.api_key == "YOUR_RANDOM_OPENAI_API_KEY_HERE":
        return "Please set your OpenAI API key in the app.py file to use this feature.", []

    # Concatenate all document content into a single context string
    context_text = "\n\n--- Document Context ---\n\n"
    for doc_id in documents_context_ids:
        content = get_full_doc_content(doc_id)
        title = get_doc_title(doc_id)
        context_text += f"Document: {title}\nContent:\n{content}\n\n"

    # Prepare messages for OpenAI, including chat history
    # The system message sets the persona and rules
    messages = [{"role": "system", "content": "You are an expert assistant for hydrogen aircraft certification. Answer questions based ONLY on the provided document context. If the answer is not in the context, state that you don't have enough information."}]
    
    # Add previous chat messages to maintain conversation context
    for msg in chat_history:
        # Ensure that only 'user' and 'assistant' roles are passed to the API
        if msg["role"] in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add the current user query with context
    messages.append({"role": "user", "content": f"Based on the following documents, answer this question: {query}\n\n{context_text}"})

    try:
        response = client.chat.completions.create( # Updated API call
            model="gpt-3.5-turbo", # You can change to "gpt-4" or other models if available
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        # Accessing content from the new response object
        answer = response.choices[0].message.content 
        return answer, documents_context_ids # Return the documents that were provided as context
    except openai.APIStatusError as e: # New error handling for API status errors
        return f"OpenAI API Error (Status {e.status_code}): {e.response}", []
    except openai.AuthenticationError: # New error handling for authentication
        return "Authentication Error: Invalid OpenAI API key. Please check your key in app.py.", []
    except openai.RateLimitError: # New error handling for rate limits
        return "Rate Limit Exceeded: You've sent too many requests. Please wait and try again.", []
    except Exception as e:
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
    st.header("Ask a Question (Chatbot)")

    # Initialize chat history in session state if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Allow user to select which documents to use as context for the query
    st.subheader("Select Documents for Context:")
    all_doc_titles = {doc['title']: doc['id'] for doc in documents_data}
    
    # Use a unique key for multiselect to avoid issues with page changes
    selected_doc_titles = st.multiselect(
        "Choose documents to provide as context to the AI:",
        options=list(all_doc_titles.keys()),
        default=[list(all_doc_titles.keys())[0]] if all_doc_titles else [],
        key="doc_context_multiselect"
    )
    selected_doc_ids = [all_doc_titles[title] for title in selected_doc_titles]

    # Chat input
    if prompt := st.chat_input("Ask a question about the selected documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.spinner("Asking AI..."):
            answer, sources_used = ask_openai_rag(prompt, selected_doc_ids, st.session_state.messages) # Pass chat history
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources_used:
                    source_titles = [get_doc_title(d_id) for d_id in sources_used]
                    st.caption(f"Sources Provided as Context: {', '.join(source_titles)}")
                else:
                    st.caption("No specific sources provided or available.")


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

    filtered_df['Related Documents'] = filtered_df['related_docs'].apply(lambda doc_ids: ", ".join([get_doc_title(d_id) for d_id in doc_ids]))
    filtered_df['Related Entities'] = filtered_df['related_entities'].apply(lambda x: ", ".join(x))

    display_cols = ['id', 'description', 'Related Documents', 'relevant_section', 'Related Entities', 'status', 'associated_discrepancy', 'notes']
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
    df_docs = pd.DataFrame(documents_data)

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
        st.write(f"**Sample Text Content:** *{content_preview[:300]}...*") # Show first 300 chars

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
        
        st.markdown("---") # Separator

elif page == "Document Comparator":
    st.header("Compare Two Documents")
    
    doc_options = {doc['title']: doc['id'] for doc in documents_data}
    
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
            st.markdown(f"**Key Topics:** {', '.join([d['key_topics'] for d in documents_data if d['id'] == doc_id_1][0])}")
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
            st.markdown(f"**Key Topics:** {', '.join([d['key_topics'] for d in documents_data if d['id'] == doc_id_2][0])}")
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

        # Updated comparison logic to use the exact titles loaded from documents_data
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

