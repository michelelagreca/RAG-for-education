# Imports
from rest_framework import generics, status
from rest_framework.response import Response
from django.http import JsonResponse
from .models import Item, Example
from .serializers import ItemSerializer, ExampleSerializer
import base64
import os
import json
import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory


# Configuration class for LangChain parameters
class LangChainConfig:
    llm_model = "gpt-4-turbo"
    chunk_size = 1000
    chunk_overlap = 200
    prompt_file = "prompt.txt"
    uploaded_file_path = './uploaded_file.pdf'
    api_key_path = './backend/GDSC/api_key.txt'
    
    @staticmethod
    def load_api_key():
        with open(LangChainConfig.api_key_path, 'r') as file:
            return file.read()


# Set an enviromental variable with the API Key of OpenAI available inside a specific textual file
def set_openai_api_key():
    """Set OpenAI API key from a file."""
    api_key = LangChainConfig.load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    print("OPENAI_API_KEY has been set!")


# Convert a PDF file to a vector store, splitting it into chunks, embedding them using OpenAI embedding model, and
# inserting it into FAISS vectore store
def pdf_to_vectorstore(pdf_path):
    
    # Open the PDF file corresponding to the document uploaded
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)

    # Convert the PDF into a textual file
    txt_file_path = 'tmp.txt'
    with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

    # Load the document
    loader = TextLoader(file_path=txt_file_path, encoding="utf-8")
    data = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=LangChainConfig.chunk_size,
        chunk_overlap=LangChainConfig.chunk_overlap
    )
    split_text = text_splitter.split_documents(data)
    
    # Embedd each chunk into a latent vector, corresponding to the semantic of the chunk
    embeddings = OpenAIEmbeddings()

    # Save the vectors of all the chunks into the FAISS vector store. 
    # They are stored and it is possible to efficiently search them into it to retrieve the embedding vectors which will be “most similar”.
    vectorstore = FAISS.from_documents(split_text, embedding=embeddings)
    return vectorstore


# Generate a conversation chain using a vector store
def gen_chain(vectorstore):

    # Initialize the LLM
    llm = ChatOpenAI(temperature=0.35, model_name=LangChainConfig.llm_model)

    # Initialize a memory element
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Initialize the conversation chain: in this phase the retriever object is defined, consisting on the vector store itself that can
    # directly retrieve the most relevant vectors based on a input (defined when the chain will be used)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Function needed to convert a plain text into a JSON.
# The idea is that the application ask always the LLM to return the answers as JSON object in order to 
# ease the usage of it in a front end. In this way, the front end has directly a JSON, that can be managed 
# easly with resepct to the rendering and the API communication.
def extract_json_from_answer(answer):
    """Extract JSON content from the GPT-4 generated answer."""
    try:
        start_idx = answer.index('{')
        end_idx = len(answer) - answer[::-1].index('}')
        return json.loads(answer[start_idx:end_idx])
    except (ValueError, json.JSONDecodeError):
        return {"error": "Failed to parse JSON from GPT-4 response."}


# Function needed to standardise the input prompt that will be used in the conversation chain.
# This apporach is used to ease the usage of the user, since she will fill a form with five fields.
# These fields are than used to create the final structured input prompt, and it correspond to the user profilation.
def create_basic_profile_prompt(profilation):
    prompt = f"""Hi! I'm {profilation['age']} years old and I'm attending {profilation["schoolOrJob"]}.
        I would like to study, 
        and in particular: {profilation['studyDescription']}.
        I commonly tend to study in a specific way, following this methodology: {profilation["methodPreference"]}.
        My goals are: {profilation["studyGoal"]}.
        These are my information on how I study and how I approach gaining new knowledge."""
    return prompt


# Class used to handle a post request coming from the front end containing the fields filled by
# the user with her information, consisting in the user profilation.
# This view returns to the front end the final structured input prompt.
class ItemCreateViewForm(generics.CreateAPIView):

    # Define the serializer for the input data received in the post request
    queryset = Item.objects.all()
    serializer_class = ItemSerializer

    def post(self, request, *args, **kwargs):

        # Serialize the data of the post request
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)

        # Save the user profile prompt consisting of the user profilation in a textual file and return back to the front end
        basic_prompt = create_basic_profile_prompt(request.data)
        with open(LangChainConfig.prompt_file, "w") as f:
            f.write(basic_prompt)

        return Response(basic_prompt, status=status.HTTP_201_CREATED)

# Class used to ask the LLM to generate a specific path related to how the user should study something.
class JourneyView(generics.CreateAPIView):

    # Define the serializer for the user profilation received in the post request
    queryset = BasicText.objects.all()
    serializer_class = BasicTextSerializer

    def post(self, request, *args, **kwargs):
        # Serialize the profilation of the post request
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)

        # Ensure the OpenAI key is set
        set_openai_api_key()

        # Initialize the vectorstore and conversation chain
        vectorstore = pdf_to_vectorstore(LangChainConfig.uploaded_file_path)
        conversation_chain = gen_chain(vectorstore)

        # Add the profilation of the user to a prompt that explain the model how to generate the study path
        query = "Profilation of the user: " + json.dumps(request.data) + "Propose a learning journey for me to help me learn about the topics in the document. Specify timeline information. Assume I have little or no prior knowledge about the topics in the document. Return the result in a JSON."

        # Ask the LLM to provide an answer based on the retrieved augmented information from the external knowledge base, and based
        # on the input prompt (prifilation + LLM guidelines)
        result = conversation_chain({"question": query})

        # Convert the answer of the LLM into a json and return back to the front end
        json_data = extract_json_from_answer(result["answer"])
        return JsonResponse(json_data, status=200)


class ExerciseView(generics.CreateAPIView):

    # Define the serializer for the profilation received in the post request
    queryset = BasicText.objects.all()
    serializer_class = BasicTextSerializer

    def post(self, request, *args, **kwargs):
        # Serialize the profilation of the post request
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)

        # Ensure the OpenAI key is set
        set_openai_api_key()

        # Initialize the vectorstore and conversation chain
        vectorstore = pdf_to_vectorstore(LangChainConfig.uploaded_file_path)
        conversation_chain = gen_chain(vectorstore)

        # Add the profilation of the user to a prompt that explain the model how to generate the study path
        query = "Profilation of the user: " + json.dumps(request.data) + "Test my knowledge about one random area among the ones identified in the map. Propose some exercise, possibly. Return the result in a JSON."
        
        # Ask the LLM to provide an answer based on the retrieved augmented information from the external knowledge base, and based
        # on the input prompt (prifilation + LLM guidelines)
        result = conversation_chain({"question": query})

        # Convert the answer of the LLM into a json and return back to the front end
        json_data = extract_json_from_answer(result["answer"])
        return JsonResponse(json_data, status=200)
