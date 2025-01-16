# Documentation Assistant

A Streamlit-based chat interface that uses AWS Bedrock to provide documentation assistance for your product.

## Features
- Chat interface for documentation queries
- Automatic query classification
- Knowledge base integration using AWS Bedrock
- Generic response handling for non-product queries

## Prerequisites
- Python 3.8+
- AWS Account with Bedrock access
- AWS Credentials with appropriate permissions
- Knowledge Base setup in AWS Bedrock

## Setup

1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Configure environment variables
- Copy `.env.example` to `.env`
```bash
cp .env.example .env
```
- Update the `.env` file with your values:
  - AWS credentials
  - AWS Region
  - Model ID
  - Knowledge Base ID
  - Product Name
  - App Title

4. Run the application
```bash
streamlit run app.py
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| AWS_REGION | AWS Region where Bedrock is configured |
| AWS_ACCESS_KEY_ID | AWS Access Key ID |
| AWS_SECRET_ACCESS_KEY | AWS Secret Access Key |
| MODEL_ID | AWS Bedrock Model ID |
| KNOWLEDGE_BASE_ID | AWS Bedrock Knowledge Base ID |
| PRODUCT_NAME | Your product name |
| APP_TITLE | Application title |

## Security Note
Never commit your `.env` file containing sensitive credentials to version control. 