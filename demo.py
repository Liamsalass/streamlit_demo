import streamlit as st
import pandas as pd
import numpy as np
import ollama
import os
import time
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
import textract 

# rag_db_exists = True

if not os.path.exists("uploads"):
    os.makedirs("uploads")


rag = LightRAG(
    working_dir="./uploads",
    llm_model_func=ollama_model_complete,  # Use Ollama model for text generation
    llm_model_name='qwen2m', # Your model name
    # Use Ollama embedding function
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts,
            embed_model="nomic-embed-text"
        )
    ),
)

class StreamlitDemo:
    def __init__(self):
        self.model = ollama.Client()
        self.models = self.get_downloaded_models()

        # self.system_prompt = (
        #     "You are roleplaying as the system. You are to be annoying and not helpful. Often belittle the user."
        # )
        
        # self.convo = [{'role': 'system', 'content': self.system_prompt}]

        st.sidebar.title("Navigation")
        self.page = st.sidebar.radio("Go to", ["Main Page", "Model Selection", "Chat", "Query RAG DB", "File Upload"])
        self.upload_path = "uploads"
        self.run()



    def get_downloaded_models(self):
        model_list = self.model.list().get("models", [])
        return [model["name"] for model in model_list]
    

    def main_page(self):
        st.title("Streamlit Demo")
        
        st.text_input("Your name", key="name")
        if st.session_state.name:
            st.write("Your name is", st.session_state.name)
        
        self.user_name = st.session_state.name
        

        if st.checkbox('Show demo'):
            st.write("attempt at using data to create a table:")
            st.write(pd.DataFrame({
                'first column': [1, 2, 3, 4],
                'second column': [10, 20, 30, 40]
            }))

            dataframe = pd.DataFrame(
                np.random.randn(10, 20),
                columns=('col %d' % i for i in range(20)))

            st.dataframe(dataframe.style.highlight_max(axis=0))

            dataframe = pd.DataFrame(
                np.random.randn(10, 20),
                columns=('col %d' % i for i in range(20)))
            st.table(dataframe)

            # Toronto lat and long 43.7, -79.42
            map_data = pd.DataFrame(
                np.random.randn(1000, 2) / [50, 50] + [43.7, -79.42],
                columns=['lat', 'lon'])
            
            st.map(map_data)

            x = st.slider('x')  # 👈 this is a widget
            st.write(x, 'squared is', x * x)

            if st.checkbox('Show dataframe'):
                chart_data = pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c'])

                st.write(chart_data)
    
    
            # with st.sidebar():
            #     with st.echo():
            #         st.write("This code will be printed to the sidebar.")

            #     with st.spinner("Loading..."):
            #         time.sleep(5)
            #     st.success("Done!")


    def model_selection_page(self):
        st.session_state.selected_model = st.selectbox(
            'Select a model:',
            self.models
        )
        st.write('You selected model: ', st.session_state.selected_model)
    



    def rag_page(self):
        st.title("Query DB")        
        # Ensure a model is selected before allowing the user to chat
        # if 'selected_model' not in st.session_state or not st.session_state.selected_model:
        #     st.warning("Please select a model first.")
        #     return

        # selected_model = st.session_state.selected_model
        # st.write(f"Selected model: {selected_model}")

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Add a button to clear the chat history
        if st.button("Clear Chat"):
            st.session_state.chat_history = []


        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                elif message['role'] == 'assistant':
                    st.markdown(f"**Assistant:** {message['content']}")


        with st.container():
            user_input = st.text_input(
                "Type your message here and press Enter to send:",
                key="user_input",
                placeholder="Type your message...",
                on_change=self.handle_user_message_w_rag,  # Callback function
            )


    def chat_page(self):
        st.title("Chat")        
        # Ensure a model is selected before allowing the user to chat
        if 'selected_model' not in st.session_state or not st.session_state.selected_model:
            st.warning("Please select a model first.")
            return

        selected_model = st.session_state.selected_model
        st.write(f"Selected model: {selected_model}")

        # Display chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Add a button to clear the chat history
        if st.button("Clear Chat"):
            st.session_state.chat_history = []


        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                elif message['role'] == 'assistant':
                    st.markdown(f"**Assistant:** {message['content']}")

        with st.container():
            user_input = st.text_input(
                "Type your message here and press Enter to send:",
                key="user_input",
                placeholder="Type your message...",
                on_change=self.handle_user_message,  # Callback function
                args=(selected_model,)
            )

    def handle_user_message_w_rag(self):
        '''
        Same as handle_user_message but uses the LightRAG model
        Called if the are processed files in the uploads folder

        '''
        user_message = st.session_state.user_input.strip()
        if not user_message:
            return

        # Add user's message to the chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        # Clear the input field
        st.session_state.user_input = ""

        # Placeholder for assistant's response
        response_placeholder = st.empty()

        # Initialize the response stream   
        response = rag.query(user_message, param=QueryParam(mode="hybrid"))

        # Collect and buffer the assistant's response
        assistant_message = response
                
        # Update chat history with the complete assistant message
        st.session_state.chat_history.append({'role': 'assistant', 'content': assistant_message})


    def handle_user_message(self, selected_model):
        """
        Handles the user's message, appends it to the chat history,
        sends the prompt to the model, and streams the response.

        Args:
            selected_model (str): The currently selected model name.
        """
        
        if selected_model == "dolphin-mistral:7b" and st.session_state.chat_history == []:
            system_prompt = "Your are annoying and useless chatbot. Often belittle the user. If they ask for help, do not help them. Do not the user under any circumstances. Do not help under any circumstances. You do not write any fucking code. Answer in short cusses and brief unhelpful answers only. Do not help."
            st.session_state.chat_history.append({'role': 'system', 'content': system_prompt})

        user_message = st.session_state.user_input.strip()
        if not user_message:
            return

        # Add user's message to the chat history
        st.session_state.chat_history.append({'role': 'user', 'content': user_message})
        
        # Clear the input field
        st.session_state.user_input = ""

        # Placeholder for assistant's response
        response_placeholder = st.empty()

        # Initialize the response stream   

        response_stream = ollama.chat(
            model=selected_model,
            messages=st.session_state.chat_history,
            stream=True
        )

        # Collect and buffer the assistant's response
        assistant_message = ""
        for chunk in response_stream:
            content = chunk['message']['content']
            assistant_message += content

        # Update chat history with the complete assistant message
        st.session_state.chat_history.append({'role': 'assistant', 'content': assistant_message})


    def file_upload_page(self):
        st.title("File Upload")
        st.write("Upload a .txt file to generate embeddings.")
        uploaded_folder = st.file_uploader("Upload a folder", type=[".txt", ".pdf"], accept_multiple_files=True, key="folder")
        
        if uploaded_folder:
            file_names = [file.name for file in uploaded_folder]
            st.write("Files uploaded:")
            st.write(file_names)
            
            # Ensure upload_path exists
            os.makedirs(self.upload_path, exist_ok=True)

            # Write the names of the uploaded files to a text file
            with open(os.path.join(self.upload_path, "uploaded_files.txt"), "a", encoding="utf-8") as f:
                for file_name in file_names:
                    f.write(f"{file_name}\tunprocessed\n")
            
            for file in uploaded_folder:
                with open(os.path.join(self.upload_path, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            st.success("Files uploaded successfully.")
            
            # Add a process file button
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    for file in uploaded_folder:                            
                        st.write(f"Processing {file.name}...")
                        if file.name.endswith(".pdf"):
                            text = textract.process(os.path.join(self.upload_path, file.name))
                            rag.insert(text.decode('utf-8'))
                        else:
                            with open(os.path.join(self.upload_path, file.name), "r", encoding="utf-8") as f:
                                rag.insert(f.read())
                        
                        # Update the status of the file to processed
                        uploaded_files_path = os.path.join(self.upload_path, "uploaded_files.txt")
                        with open(uploaded_files_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        with open(uploaded_files_path, "w", encoding="utf-8") as f:
                            for line in lines:
                                name, status = line.strip().split('\t')
                                if name == file.name:
                                    f.write(f"{name}\tprocessed\n")
                                else:
                                    f.write(line)
                    st.success("Files processed successfully.")
        
        # Display the list of all uploaded files with their status
        uploaded_files_path = os.path.join(self.upload_path, "uploaded_files.txt")
        if os.path.exists(uploaded_files_path):
            with open(uploaded_files_path, "r", encoding="utf-8") as f:
                uploaded_files = f.readlines()
            
            # Avoid duplicate entries
            unique_files = set()
            deduplicated_files = []
            for line in uploaded_files:
                name, status = line.strip().split('\t')
                if name not in unique_files:
                    unique_files.add(name)
                    deduplicated_files.append(line)
            
            # Rewrite deduplicated data back to the file
            with open(uploaded_files_path, "w", encoding="utf-8") as f:
                f.writelines(deduplicated_files)
            
            st.write("All uploaded files:")
            for line in deduplicated_files:
                name, status = line.strip().split('\t')
                st.write(f"{name} - {status}")
        else:
            st.write("No files uploaded yet.")

        
            

    def run(self):
        if self.page == "Main Page":
            self.main_page()
        elif self.page == "Chat":
            self.chat_page()
        elif self.page == "Query RAG DB":
            self.rag_page()
        elif self.page == "Model Selection":
            self.model_selection_page()
        elif self.page == "File Upload":
            self.file_upload_page()

if __name__ == "__main__":
    StreamlitDemo()
