from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)
from langchain.embeddings import HuggingFaceEmbeddings
import math
from datetime import datetime, timedelta

import faiss

from llm import falcon_llm
# LLM = ChatOpenAI(max_tokens=1500)


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    # embeddings_model = OpenAIEmbeddings()
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


tommies_memory = GenerativeAgentMemory(
    llm=falcon_llm,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

tommie = GenerativeAgent(
    name="Tommie",
    age=25,
    traits="anxious, likes design, talkative",  # You can add more persistent traits here
    status="looking for a job",  # When connected to a virtual world, we can have the characters update their status
    memory_retriever=create_new_memory_retriever(),
    llm=falcon_llm,
    memory=tommies_memory,
)

eves_memory = GenerativeAgentMemory(
    llm=falcon_llm,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=5,
)


eve = GenerativeAgent(
    name="Eve",
    age=34,
    traits="curious, helpful",  # You can add more persistent traits here
    status="N/A",  # When connected to a virtual world, we can have the characters update their status
    llm=falcon_llm,
    daily_summaries=[
        (
            "Eve started her new job as a career counselor last week and received her first assignment, a client named Tommie."
        )
    ],
    memory=eves_memory,
    verbose=False,
)


if "__name__" == "__main__":
    # The current "Summary" of a character can't be made because the agent hasn't made
    # any observations yet.
    print(tommie.get_summary())

    # We can add memories directly to the memory object
    tommie_observations = [
        "Tommie remembers his dog, Bruno, from when he was a kid",
        "Tommie feels tired from driving so far",
        "Tommie sees the new home",
        "The new neighbors have a cat",
        "The road is noisy at night",
        "Tommie is hungry",
        "Tommie tries to get some rest.",
    ]
    for observation in tommie_observations:
        tommie.memory.add_memory(observation)

    print(tommie.get_summary())

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%A %B %d")
    eve_observations = [
        "Eve wakes up and hear's the alarm",
        "Eve eats a boal of porridge",
        "Eve helps a coworker on a task",
        "Eve plays tennis with her friend Xu before going to work",
        "Eve overhears her colleague say something about Tommie being hard to work with",
    ]
    for observation in eve_observations:
        eve.memory.add_memory(observation)

    print(eve.get_summary())