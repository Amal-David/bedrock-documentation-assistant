import boto3
import streamlit as st
import json
from dotenv import load_dotenv
import os
import logging
import traceback
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Validate required environment variables
required_env_vars = [
    'AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY',
    'MODEL_ID', 'KNOWLEDGE_BASE_ID', 'PRODUCT_NAME', 'APP_TITLE'
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise ValueError(error_msg)

st.title(os.getenv('APP_TITLE'))
st.subheader("Chat Interface", divider='rainbow')

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    logger.info("Chat history initialized")

# Initialize Bedrock clients
try:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    logger.info("Bedrock Runtime client initialized successfully")

    bedrock_agent = boto3.client(
        service_name='bedrock-agent-runtime',
        region_name=os.getenv('AWS_REGION'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    logger.info("Bedrock Agent Runtime client initialized successfully")
except Exception as e:
    error_msg = f"Failed to initialize Bedrock clients: {str(e)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    st.error(error_msg)

def classify_query(query: str) -> str:
    """
    Classify if the query is product-specific or generic using the Amazon Nova Chat model.
    """
    logger.info(f"Classifying query: {query}")

    try:
        system_list = [
            {
                "text": f"""Classify user input into:
"Product" - for {os.getenv('PRODUCT_NAME')} specific queries
"Generic" - for general questions.
Only respond with category.
Just Product or Generic, nothing more or less."""
            }
        ]

        message_list = [{"role": "user", "content": [{"text": query}]}]

        inf_params = {
            "max_new_tokens": 10,
            "top_p": 0.9,
            "top_k": 20,
            "temperature": 0.7
        }
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params,
        }

        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body).encode('utf-8'),
            modelId=os.getenv('MODEL_ID'),
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        logger.debug(f"Classification response: {json.dumps(response_body, indent=2)}")

        content = response_body['output']['message']['content']
        result = ''.join(item['text'] for item in content if 'text' in item)

        classification = "Product" if os.getenv('PRODUCT_NAME').lower() in result.lower() else "Generic"
        logger.info(f"Query classified as: {classification}")
        return classification

    except Exception as e:
        error_msg = f"Classification error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        return "Generic"

def get_kb_response(query: str) -> str:
    """
    Get a response using the Knowledge Base (via bedrock-agent's 'retrieve_and_generate').
    """
    logger.info(f"Getting knowledge base response for query: {query}")
    try:
        kb_response = bedrock_agent.retrieve_and_generate(
            input={'text': query},
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': os.getenv('KNOWLEDGE_BASE_ID'),
                    'modelArn': f"arn:aws:bedrock:{os.getenv('AWS_REGION')}::foundation-model/{os.getenv('MODEL_ID')}"
                }
            }
        )

        logger.debug(f"Knowledge base response: {json.dumps(kb_response, indent=2)}")
        answer = kb_response['output']['text']
        logger.info("Successfully retrieved answer from knowledge base")
        
        if (len(kb_response['citations']) > 0 and
            len(kb_response['citations'][0]['retrievedReferences']) > 0):
            context = kb_response['citations'][0]['retrievedReferences'][0]['content']['text']
            source = kb_response['citations'][0]['retrievedReferences'][0]['location']['s3Location']['uri']
            logger.info(f"Found context from source: {source}")
            st.markdown(
                f"<span style='color:#FFDA33'>Context: </span>{context}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<span style='color:#FFDA33'>Source: </span>{source}",
                unsafe_allow_html=True
            )
        else:
            logger.warning("No specific context found in knowledge base response")
            st.markdown("<span style='color:red'>No specific context found</span>", unsafe_allow_html=True)
            
        return answer

    except Exception as e:
        error_msg = f"Knowledge base error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        return ""

def get_generic_response(query: str) -> str:
    """
    Get a generic response using the Amazon Nova Chat model.
    """
    logger.info(f"Getting generic response for query: {query}")
    try:
        system_list = [
            {
                "text": "You are a helpful assistant. Please provide a clear, professional answer."
            }
        ]

        message_list = [{"role": "user", "content": [{"text": query}]}]

        inf_params = {
            "max_new_tokens": 500,
            "top_p": 0.9,
            "top_k": 20,
            "temperature": 0.7
        }
        request_body = {
            "schemaVersion": "messages-v1",
            "messages": message_list,
            "system": system_list,
            "inferenceConfig": inf_params,
        }

        response = bedrock_runtime.invoke_model(
            body=json.dumps(request_body).encode('utf-8'),
            modelId=os.getenv('MODEL_ID'),
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response['body'].read())
        logger.debug(f"Generic response body: {json.dumps(response_body, indent=2)}")

        content = response_body['output']['message']['content']
        result = ''.join(item['text'] for item in content if 'text' in item)
        logger.info("Successfully generated generic response")
        return result

    except Exception as e:
        error_msg = f"Generic response error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        st.error(error_msg)
        return ""

if query := st.chat_input(f"Ask me anything about {os.getenv('PRODUCT_NAME')}..."):
    logger.info(f"New query received: {query}")

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})
    logger.debug("User message added to chat history")

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            category = classify_query(query)
            st.session_state.chat_history.append(
                {"role": "system", "content": f"Classified as: {category}"}
            )
            logger.info(f"Query classification added to chat history: {category}")

            if category == "Product":
                logger.info("Processing product-specific query")
                response = get_kb_response(query)
            else:
                logger.info("Processing generic query")
                response = get_generic_response(query)

            if response:
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                logger.info("Assistant response added to chat history")
            else:
                logger.warning("No response generated")

with st.sidebar:
    st.header("Configuration")
    st.write("Region:", os.getenv('AWS_REGION'))
    st.write("Model:", os.getenv('MODEL_ID'))
    st.write("Product:", os.getenv('PRODUCT_NAME'))
    
    if st.button("Clear Chat"):
        logger.info("Clearing chat history")
        st.session_state.chat_history = []
        st.rerun()