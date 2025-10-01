"""LangGraph RAG pipeline with Qdrant vector store."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langchain.schema import Document
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Get environment variables at module level to avoid blocking calls in async functions
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "learn_vector")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class Context(TypedDict):
    """Context parameters for the RAG agent."""

    openai_api_key: str
    qdrant_url: str
    collection_name: str
    max_tokens: int
    temperature: float


@dataclass
class State:
    """State for the RAG pipeline."""

    query: str
    query_components: List[str] = None
    component_results: Dict[str, List[Document]] = None
    fused_documents: List[Document] = None
    context: str = ""
    answer: str = ""
    error: str = ""
    iteration_count: int = 0
    max_iterations: int = 4
    is_relevant: bool = False
    missing_information: str = ""
    failed_components: List[str] = None


async def decompose_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Decompose complex queries into semantic components."""
    try:
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "query_components": [state.query],
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM for query decomposition
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.2,  # Low temperature for consistent decomposition
            max_tokens=300
        )
        
        # Create prompt template for query decomposition
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a query decomposition expert for insurance policy queries. Your task is to break down complex queries into multiple complete, searchable queries that can be searched independently.

Instructions:
- Break the query into 2-4 complete, focused queries
- Each decomposed query should be a complete, searchable question
- Focus on different aspects: medical procedures, geographic coverage, policy terms, conditions
- Make each query specific and comprehensive
- Each query should be able to find relevant information independently

Examples:
- "eye surgery in Dubai" → ["What is covered for eye surgery procedures?", "What coverage is available for medical treatment in Dubai or UAE?", "What are the international medical coverage terms?"]
- "car accident coverage" → ["What coverage is provided for car accidents?", "What are the collision coverage terms?", "What auto insurance benefits are available?"]
- "dental implants cost" → ["What dental procedures are covered?", "What are the costs and expenses for dental treatment?", "What dental coverage benefits are available?"]

Return the decomposed queries as a JSON list of complete questions."""),
            ("human", "Decompose this insurance query into multiple complete searchable queries: {query}")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate components
        components_response = await chain.ainvoke({
            "query": state.query
        })
        
        # Parse JSON response
        import json
        try:
            components = json.loads(components_response)
            if not isinstance(components, list):
                components = [components]
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            components = [state.query]
        
        logger.info(f"Decomposed query '{state.query}' into components: {components}")
        
        return {
            "query_components": components,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error decomposing query: {str(e)}")
        return {
            "query_components": [state.query],  # Fallback to original query
            "error": f"Error decomposing query: {str(e)}"
        }


async def enhance_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Enhance the user query for better retrieval from vector database."""
    try:
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "enhanced_query": state.query,
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM for query enhancement
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.3,  # Lower temperature for more focused query enhancement
            max_tokens=200
        )
        
        # Create prompt template for query enhancement
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a query enhancement assistant. Your task is to improve user queries to get better results from a document retrieval system.

Your goal is to:
1. Expand the query with relevant synonyms and related terms
2. Add context that might help find relevant documents
3. Include alternative phrasings of the same concept
4. Make the query more specific and detailed for better retrieval
5. Keep the enhanced query concise but comprehensive

Original query: {original_query}

Return only the enhanced query, no explanations or additional text."""),
            ("human", "Enhance this insurance policy query for better document retrieval: {original_query}")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate enhanced query
        enhanced_query = await chain.ainvoke({
            "original_query": state.query
        })
        
        logger.info(f"Enhanced query: '{state.query}' -> '{enhanced_query}'")
        
        return {
            "enhanced_query": enhanced_query,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error enhancing query: {str(e)}")
        return {
            "enhanced_query": state.query,  # Fallback to original query
            "error": f"Error enhancing query: {str(e)}"
        }


async def parallel_retrieve_documents(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Retrieve documents for each query component in parallel."""
    try:
        if not state.query_components:
            logger.warning("No query components to retrieve")
            return {
                "component_results": {},
                "error": "No query components available"
            }
        
        # Import vector store here to avoid circular imports
        from src.agent.simple_vector_store import SimpleVectorStore
        
        # Initialize vector store using module-level environment variables
        vector_store = SimpleVectorStore(
            collection_name=QDRANT_COLLECTION,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        
        component_results = {}
        
        # Search each decomposed query separately
        for i, component_query in enumerate(state.query_components):
            try:
                logger.info(f"Searching for decomposed query {i+1}: '{component_query}'")
                search_result = vector_store.similarity_search(query=component_query, k=3)
                component_results[component_query] = search_result
                logger.info(f"Found {len(search_result)} documents for query: '{component_query}'")
            except Exception as component_error:
                logger.error(f"Error searching component query '{component_query}': {str(component_error)}")
                component_results[component_query] = []
        
        return {
            "component_results": component_results,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error in parallel retrieval: {str(e)}")
        return {
            "component_results": {},
            "error": f"Error in parallel retrieval: {str(e)}"
        }


async def retrieve_documents(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Retrieve documents from vector store."""
    try:
        # Import vector store here to avoid circular imports
        from src.agent.simple_vector_store import SimpleVectorStore
        
        # Initialize vector store using module-level environment variables
        vector_store = SimpleVectorStore(
            collection_name=QDRANT_COLLECTION,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Use enhanced query if available, otherwise fallback to original query
        search_query = state.enhanced_query if state.enhanced_query else state.query
        
        # Search for relevant documents
        search_result = vector_store.similarity_search(query=search_query, k=5)
        
        if not search_result:
            logger.warning(f"No relevant documents found for query: {state.query}")
            return {
                "retrieved_documents": [],
                "error": "No relevant information found in the uploaded documents."
            }
        
        logger.info(f"Retrieved {len(search_result)} documents for query: {state.query}")
        
        return {
            "retrieved_documents": search_result,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        return {
            "retrieved_documents": [],
            "error": f"Error retrieving documents: {str(e)}"
        }


async def judge_relevance(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Judge if the retrieved context is relevant to the query."""
    try:
        if not state.context:
            logger.warning("No context to judge relevance")
            return {
                "is_relevant": False,
                "error": "No context available to judge relevance"
            }
        
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "is_relevant": False,
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM for relevance judgment
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.1,  # Very low temperature for consistent judgment
            max_tokens=100
        )
        
        # Create prompt template for relevance judgment
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a strict relevance judge for insurance policy queries. Your task is to determine if the retrieved context contains SPECIFIC information that can directly answer the user's question.

Instructions:
- Be VERY STRICT - only mark as relevant if the context contains SPECIFIC information that answers the question
- Look for exact details, specific coverage information, terms, conditions, or amounts that directly relate to the question
- If the context is about the same topic but doesn't contain the specific information needed, mark as NOT relevant
- If the context is too general or doesn't have the specific details, mark as NOT relevant
- Only say YES if you can find the exact information the user is asking for
- Analyze the context from multiple search components to see if they collectively answer the question

Format your response as:
YES - if the context contains specific information to answer the question
NO: [specific information that is missing] - if the context doesn't contain the needed information

Example: "NO: The context doesn't contain information about coverage for eye surgery abroad or international medical procedures" """),
            ("human", """User Question: {question}

Context:
{context}

Does this context contain SPECIFIC information that can directly answer the user's question? Answer YES or NO with missing information details.""")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate judgment
        judgment = await chain.ainvoke({
            "question": state.query,
            "context": state.context
        })
        
        # Parse judgment and extract missing information
        judgment_clean = judgment.strip().upper()
        is_relevant = judgment_clean.startswith("YES")
        
        missing_information = ""
        failed_components = []
        
        if not is_relevant and "NO:" in judgment:
            # Extract missing information after "NO:"
            missing_part = judgment.split("NO:", 1)[1].strip()
            missing_information = missing_part
            
            # Identify which query components might have failed
            # This is a simple heuristic - you can make it more sophisticated
            if state.query_components:
                for component_query in state.query_components:
                    # Check if the missing information relates to this component
                    component_keywords = component_query.lower().split()
                    missing_keywords = missing_part.lower().split()
                    
                    # If there's overlap between component and missing info, mark as failed
                    if any(keyword in missing_keywords for keyword in component_keywords if len(keyword) > 3):
                        failed_components.append(component_query)
        elif not is_relevant:
            # Fallback if format is different
            missing_information = "Context does not contain specific information to answer the question"
            failed_components = state.query_components.copy() if state.query_components else []
        
        logger.info(f"Relevance judgment: {judgment} (Relevant: {is_relevant})")
        if missing_information:
            logger.info(f"Missing information: {missing_information}")
        if failed_components:
            logger.info(f"Failed components: {failed_components}")
        
        return {
            "is_relevant": is_relevant,
            "missing_information": missing_information,
            "failed_components": failed_components,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error judging relevance: {str(e)}")
        return {
            "is_relevant": False,
            "missing_information": "Error in relevance judgment",
            "failed_components": state.query_components.copy() if state.query_components else [],
            "error": f"Error judging relevance: {str(e)}"
        }


async def rewrite_failed_components(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Rewrite only the failed query components."""
    try:
        if not state.failed_components:
            logger.warning("No failed components to rewrite")
            return {
                "query_components": state.query_components,
                "error": "No failed components to rewrite"
            }
        
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "query_components": state.query_components,
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM for component rewriting
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5",
            temperature=0.3,
            max_tokens=200
        )
        
        # Create prompt template for component rewriting
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a query component rewriting expert for insurance policy queries. Your task is to rewrite only the failed query components while keeping successful ones unchanged.

Instructions:
- Only rewrite the query components that failed to retrieve relevant information
- Keep successful query components exactly the same
- Rewrite failed components as complete, searchable questions
- Use different terminology, synonyms, and related concepts for failed components
- Focus on the specific aspect that failed
- Make the rewritten components more targeted and specific
- Each rewritten component should be a complete question

Original query: {original_query}
Failed components: {failed_components}
Current components: {current_components}
Missing information: {missing_information}

Rewrite only the failed components as complete questions, keeping successful ones unchanged."""),
            ("human", "Rewrite the failed query components as complete questions: {failed_components}")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate rewritten components
        rewritten_response = await chain.ainvoke({
            "original_query": state.query,
            "failed_components": state.failed_components,
            "current_components": state.query_components,
            "missing_information": state.missing_information
        })
        
        # Parse the response and update only failed components
        import json
        try:
            new_components = json.loads(rewritten_response)
            if not isinstance(new_components, list):
                new_components = [new_components]
        except json.JSONDecodeError:
            # Fallback: rewrite failed components manually
            new_components = state.query_components.copy()
            for i, component in enumerate(new_components):
                if component in state.failed_components:
                    # Simple rewriting for failed components
                    new_components[i] = f"{component} insurance coverage policy terms"
        
        logger.info(f"Rewritten failed components: {state.failed_components} -> {new_components}")
        
        return {
            "query_components": new_components,
            "iteration_count": state.iteration_count + 1,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error rewriting failed components: {str(e)}")
        return {
            "query_components": state.query_components,
            "iteration_count": state.iteration_count + 1,
            "error": f"Error rewriting failed components: {str(e)}"
        }


async def generate_new_enhanced_query(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate a new enhanced query when context is not relevant."""
    try:
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "enhanced_query": state.query,
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM for query enhancement
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.4,  # Slightly higher temperature for more creative enhancement
            max_tokens=200
        )
        
        # Create prompt template for new query enhancement
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a query enhancement assistant specialized in insurance policy queries. The previous enhanced query did not retrieve relevant context, so you need to create a completely different approach.

Your task is to:
1. Analyze the original query and the missing information identified by the judge
2. Create a new enhanced query that specifically targets the missing information
3. Use different insurance terminology, synonyms, and related concepts
4. Focus on the specific aspects mentioned in the missing information
5. Include alternative phrasings and related insurance terms that might find the missing information
6. Make the query more targeted to find the specific missing details

Original query: {original_query}
Previous enhanced query: {previous_enhanced_query}
Missing information: {missing_information}
Current context (not relevant): {current_context}

Create a new enhanced query that specifically targets finding the missing information: {missing_information}"""),
            ("human", "Generate a new enhanced query for: {original_query} focusing on finding: {missing_information}")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate new enhanced query
        new_enhanced_query = await chain.ainvoke({
            "original_query": state.query,
            "previous_enhanced_query": state.enhanced_query,
            "missing_information": state.missing_information,
            "current_context": state.context[:500] if state.context else "No context available"
        })
        
        logger.info(f"Generated new enhanced query: '{state.query}' -> '{new_enhanced_query}'")
        
        return {
            "enhanced_query": new_enhanced_query,
            "iteration_count": state.iteration_count + 1,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error generating new enhanced query: {str(e)}")
        return {
            "enhanced_query": state.query,  # Fallback to original query
            "iteration_count": state.iteration_count + 1,
            "error": f"Error generating new enhanced query: {str(e)}"
        }


async def semantic_fusion(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Intelligently combine results from parallel component searches."""
    try:
        if not state.component_results:
            logger.warning("No component results to fuse")
            return {
                "fused_documents": [],
                "error": "No component results available"
            }
        
        # Collect all documents from all components
        all_documents = []
        for component, documents in state.component_results.items():
            for doc in documents:
                # Add component metadata to each document
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['search_component'] = component
                all_documents.append(doc)
        
        # Remove duplicates based on page_content
        seen_content = set()
        unique_documents = []
        for doc in all_documents:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_documents.append(doc)
        
        # Sort by relevance (you can implement more sophisticated ranking here)
        # For now, we'll keep the order as retrieved
        fused_documents = unique_documents
        
        logger.info(f"Fused {len(all_documents)} total documents into {len(fused_documents)} unique documents")
        
        return {
            "fused_documents": fused_documents,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error in semantic fusion: {str(e)}")
        return {
            "fused_documents": [],
            "error": f"Error in semantic fusion: {str(e)}"
        }


async def generate_context(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate context from fused documents."""
    try:
        # Use fused_documents if available, otherwise fallback to retrieved_documents
        documents_to_use = state.fused_documents if state.fused_documents else state.retrieved_documents
        
        if not documents_to_use:
            logger.warning("No documents to generate context from")
            return {
                "context": "",
                "error": "No documents available to generate context"
            }
        
        # Format context with page_content, page_label, and search component
        context_parts = []
        for doc in documents_to_use:
            page_label = doc.metadata.get('page_label', 'Unknown')
            search_component = doc.metadata.get('search_component', 'Unknown')
            context_parts.append(
                f"page_content: {doc.page_content}\n"
                f"page_label: {page_label}\n"
                f"search_component: {search_component}"
            )
        
        context = "\n\n\n".join(context_parts)
        
        logger.info(f"Generated context for query: {state.query}")
        
        return {
            "context": context,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error generating context: {str(e)}")
        return {
            "context": "",
            "error": f"Error generating context: {str(e)}"
        }


async def generate_answer(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Generate answer using LLM."""
    try:
        # Debug: Log the state context
        logger.info(f"State context: {state.context}")
        
        # Use module-level environment variable
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found in environment variables")
            return {
                "answer": "Error: OpenAI API key not provided",
                "error": "OpenAI API key not provided"
            }
        
        # Create LLM with module-level environment variable
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful insurance policy assistant that answers questions based on the provided context.

Context:
{context}

Instructions:
- Use the information provided in the context to answer the question as best as possible
- ALWAYS include the page number (page_label) for every piece of information you mention
- If the context contains related information but not the exact answer, provide what information is available and explain what's missing
- If the context doesn't contain the specific information, explain what information is available and suggest where to find more details
- Be helpful by providing any related information that might be useful
- Be concise and accurate in your response
- Format your answer with clear references to page numbers
- If you're unsure about something, mention that you're not certain

Example format: "According to Page 5, your policy covers..." or "As stated on Page 12, the deductible is..." or "While Page 3 mentions general coverage, specific details about [topic] are not provided in the available context."""),
            ("human", "Question: {question}")
        ])
        
        # Create chain
        chain = prompt_template | llm | StrOutputParser()
        
        # Generate answer
        answer = await chain.ainvoke({
            "context": state.context,
            "question": state.query
        })
        
        logger.info(f"Generated answer for query: {state.query}")
        
        return {
            "answer": answer,
            "error": ""
        }
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return {
            "answer": f"Error generating answer: {str(e)}",
            "error": f"Error generating answer: {str(e)}"
        }


async def should_continue_after_judge(state: State, runtime: Runtime[Context]) -> str:
    """Determine next step after relevance judgment."""
    if state.error:
        return "error"
    
    if state.is_relevant:
        logger.info("Context is relevant, proceeding to generate answer")
        return "generate_answer"
    elif state.iteration_count >= state.max_iterations:
        logger.info(f"Max iterations ({state.max_iterations}) reached, proceeding with current context")
        return "generate_answer"
    else:
        logger.info(f"Iteration {state.iteration_count + 1}: Context not relevant, rewriting failed components")
        return "rewrite_failed_components"


async def should_continue_after_enhance(state: State, runtime: Runtime[Context]) -> str:
    """Determine if we should continue processing after query enhancement."""
    if state.error:
        return "error"
    return "retrieve"


# Define the graph
graph = (
    StateGraph(State)
    .add_node("decompose_query", decompose_query)
    .add_node("parallel_retrieve", parallel_retrieve_documents)
    .add_node("semantic_fusion", semantic_fusion)
    .add_node("generate_context", generate_context)
    .add_node("judge_relevance", judge_relevance)
    .add_node("rewrite_failed_components", rewrite_failed_components)
    .add_node("generate_answer", generate_answer)
    .add_edge("__start__", "decompose_query")
    .add_edge("decompose_query", "parallel_retrieve")
    .add_edge("parallel_retrieve", "semantic_fusion")
    .add_edge("semantic_fusion", "generate_context")
    .add_edge("generate_context", "judge_relevance")
    .add_conditional_edges(
        "judge_relevance",
        should_continue_after_judge,
        {
            "generate_answer": "generate_answer",
            "rewrite_failed_components": "rewrite_failed_components",
            "error": END
        }
    )
    .add_conditional_edges(
        "rewrite_failed_components",
        should_continue_after_enhance,
        {
            "parallel_retrieve": "parallel_retrieve",
            "error": END
        }
    )
    .add_edge("generate_answer", END)
    .compile(name="RAG Pipeline")
)
