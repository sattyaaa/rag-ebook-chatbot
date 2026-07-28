import streamlit as st
from rag.graph import rag_graph
from rag.ingest import ingest_pdf

# Set page configuration
st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic AI eBook RAG Chatbot")
st.markdown("Ask any questions about the Agentic AI eBook. The chatbot will retrieve facts directly from the document.")

# Sidebar for PDF Ingestion
st.sidebar.title("Document Manager")
st.sidebar.markdown("Process and embed the ebook PDF into Pinecone.")

if st.sidebar.button("🔄 Ingest PDF", use_container_width=True):
    with st.sidebar.spinner("Processing & indexing PDF..."):
        try:
            total_chunks = ingest_pdf(reset_namespace=True)
            st.sidebar.success(f"Successfully Ingested {total_chunks} chunks!")
        except Exception as e:
            st.sidebar.error(f"Error during ingestion: {str(e)}")

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("chunks"):
            st.caption("🔍 **Retrieved Context Chunks (Hover to view content):**")
            chunks_to_show = message["chunks"][:10]
            cols = st.columns(len(chunks_to_show))
            for i, chunk in enumerate(chunks_to_show):
                score = chunk.get("score", 0.0)
                source = chunk.get("source", "Unknown")
                page = chunk.get("page")
                page_str = f"P. {page + 1}" if page is not None else "P. N/A"
                tooltip_text = f"**Chunk {i+1}**\n\n**Score:** {score:.4f}\n\n**Source:** {source}\n\n**Content:**\n{chunk.get('text')}"
                with cols[i]:
                    st.button(
                        label=page_str,
                        help=tooltip_text,
                        key=f"hist_btn_{msg_idx}_{i}",
                        use_container_width=True
                    )

# User Chat Input
if query := st.chat_input("Ask a question about the ebook..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Process via LangGraph RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag_graph.invoke({"question": query})
                answer = result.get("final_answer", "I don't know based on the provided PDF. What can I help you with? Please ask questions from the PDF.")
                chunks = result.get("chunks", [])
                
                # Render response
                st.write(answer)
                
                if chunks:
                    st.caption("🔍 **Retrieved Context Chunks (Hover to view content):**")
                    chunks_to_show = chunks[:10]
                    cols = st.columns(len(chunks_to_show))
                    for i, chunk in enumerate(chunks_to_show):
                        score = chunk.get("score", 0.0)
                        source = chunk.get("source", "Unknown")
                        page = chunk.get("page")
                        page_str = f"P. {page + 1}" if page is not None else "P. N/A"
                        tooltip_text = f"**Chunk {i+1}**\n\n**Score:** {score:.4f}\n\n**Source:** {source}\n\n**Content:**\n{chunk.get('text')}"
                        with cols[i]:
                            st.button(
                                label=page_str,
                                help=tooltip_text,
                                key=f"new_btn_{i}",
                                use_container_width=True
                            )
                
                # Save to session history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks
                })
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
