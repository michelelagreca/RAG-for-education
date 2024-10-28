# Educational RAG System with LangChain & Django

This project is a Django-based backend designed to leverage Retrieval-Augmented Generation (RAG) for educational purposes. By integrating LangChain, FAISS for vector storage, and OpenAI's ChatGPT model, this system creates customized learning journeys and exercises tailored to user profiles. The responses generated in this backend are served to a React frontend for user interaction. React is not part of the implementation of this project.

## Key Features

The key feature of the backend are placed in the `view.py` file of the GDSC folder.
- **Custom Study Paths**: Generates personalized study paths based on user profiles.
- **Dynamic Exercises**: Provides exercises and quizzes for assessing user knowledge.
- **User Profiling**: Collects user preferences and goals to customize learning content.
- **Efficient Knowledge Retrieval**: Uses FAISS vector storage for efficient semantic search across uploaded documents.
- **Conversational Memory**: Maintains conversation history to support context-aware interactions with users.

## Tech Stack
- **Backend**: Django, Django REST Framework, LangChain, FAISS, OpenAI ChatGPT APIs
- **Frontend**: React (not included in this repository)
- **LLM**: OpenAI's ChatGPT
- **Vector Store**: FAISS for efficient semantic retrieval of document chunks

---

## Backend Workflow
1. **User Profile Submission**: The frontend sends a POST request to the backend endpoint `item/form/` with user profiling data (age, study goals, methods, etc.).
2. **PDF Document Processing**: A PDF containing the external knowledge base is processed, split into chunks, embedded, and stored in a FAISS vector store.
3. **Context Generation**: Using the user profile and FAISS-powered knowledge retrieval, a context is generated. By calling endpoints `journey/` and `exercises/` a customized study journey or exercise prompt is generated.
4. **LLM Queryn and Conversation Chain**: The ChatGPT model is used to generate study paths or exercises based on user-specific prompts and the document's content by calling endpoints `journey/` and `exercises/`.
5. **JSON Output**: Responses from the model are converted into JSON format for ease of use by the frontend.

---


## API Endpoints

1. **`/item/form`**: Accepts user profiling data, creates a structured prompt, and saves it for subsequent use.
2. **`/journey`**: Uses profiling data and document content to generate a customized learning journey.
3. **`/exercise`**: Creates exercises based on the user's profile and document content for knowledge assessment.

---

## Notebook examples

In the notebook `examples.ipynb` it is possible to find some example and the relative output of the LLM regarding various scenarios.


## Contributing

The project has been developed together with Alessandro Mileto, Nicholas Nicolis and Vincenzo Martello as part of the challenge *GDSC Polimi - AI Hack*.
