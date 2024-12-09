import os
import re
import json
import openai
import concurrent
import tiktoken
import asyncio
import PyPDF2
import heapq
import faiss
import numpy as np
import pandas as pd
from copy import deepcopy
from dotenv import load_dotenv
from tqdm.notebook import tqdm
from typing import List


load_dotenv()

# Add in your Open AI Key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()


def extract_icd_codes(text):
    """
    Extracts ICD codes from a string.
    """
    icd_pattern = r"\b([A-Z]\d{2}(?:\.\d{1,4})?)\b"
    icd_codes = re.findall(icd_pattern, text)
    return icd_codes


def extract_cpt_codes(text):
    """
    Extracts CPT codes from a string.
    """
    cpt_pattern = r"\b(\d{5}[A-Z]?|\d{4}[A-Z])\b"
    cpt_codes = re.findall(cpt_pattern, text)
    return cpt_codes


def extract_hcpcs_codes(text):
    """
    Extracts HCPCS codes from a string.
    """
    hcpcs_pattern = r"\b([A-Z]\d{4}[A-Z]?)\b"
    hcpcs_codes = re.findall(hcpcs_pattern, text)

    return hcpcs_codes


def extract_codes(text):
    """
    Extracts ICD, CPT, and HCPCS codes from a string.

    Args:
        text (str): The input string.

    Returns:
        list: A list containing ICD, CPT, and HCPCS codes.

    """
    icd_codes = extract_icd_codes(text)
    cpt_codes = extract_cpt_codes(text)
    hcpcs_codes = extract_hcpcs_codes(text)
    codes = icd_codes + cpt_codes + hcpcs_codes
    return codes


def cosine_similarity_faiss(x, y):
    # Convert the inputs to float32 and reshape if necessary
    x = np.ascontiguousarray(x).astype("float32")
    y = np.ascontiguousarray(y).astype("float32")

    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    if len(y.shape) == 1:
        y = y.reshape(1, -1)

    # Normalize the vectors
    faiss.normalize_L2(x)
    faiss.normalize_L2(y)

    # Build the index
    index = faiss.IndexFlatIP(y.shape[1])
    index.add(y)

    # Compute the cosine similarity
    _, similarity = index.search(x, y.shape[0])

    return similarity


def eval_accuracy(results_file, eval_by_category=True, print_overall_accuracy=True):
    """
    Calculate and print the accuracy of the model's chosen answers.

    Args:
        results_file (str): The path to the CSV file containing the results.
        eval_by_category (bool): boolean, True if accuracy should be evaluated by category
    Returns:
        results_df (pd.DataFrame): The DataFrame containing the results.
    """
    results = pd.read_csv(results_file)

    def _extract_option_letter(row):
        """
        Extracts the option letter from the model answer in the given row.
        """
        model_answer = ""
        try:
            if "ANSWER" in row["model_answer"]:
                model_answer = row["model_answer"].split("ANSWER")[1]
            elif "Answer" in row["model_answer"]:
                model_answer = row["model_answer"].split("Answer")[1]
        except:
            print(row["model_answer"])
            print("******")

        # print(model_answer)
        options = re.findall(r"\b([a-dA-D])\.", model_answer)
        if options and len(options) == 1:
            return options[0].lower()
        else:
            # TODO: Check cases where options list is longer, should be handled with the prompt
            # print(options, model_answer)
            return None

    results["model_chosen_answer_letter"] = results.apply(
        _extract_option_letter, axis=1
    )
    results["correct"] = np.where(
        results["model_chosen_answer_letter"] == results["answer_letter"], 1, 0
    )
    overall_accuracy = np.round(results["correct"].sum() / len(results) * 100, 2)
    if print_overall_accuracy:
        print(f"Overall Accuracy: {overall_accuracy}%\n")
    if eval_by_category:
        _eval_by_category(results)

    return overall_accuracy, results


def search_relevant_strings(query, filtered_df, n):
    """
    Search for the top n most relevant strings in a dataframe based on a query string.

    Args:
        query (str): The query string to search for.
        df (pandas.DataFrame): The dataframe containing the strings to search in.
        n (int): The number of top relevant strings to return.

    Returns:
        list: A list of the top n most relevant strings.
    """

    # Get the embeddings for the query string
    query_embedding = np.array(get_embeddings([query]))

    # Check if the embeddings column exists in the dataframe
    if "embeddings" not in filtered_df.columns:
        raise ValueError("The 'embeddings' column does not exist in the dataframe.")

    # Get the embeddings for the combined strings in the dataframe
    combined_embeddings = np.array(filtered_df["embeddings"].tolist())

    # Calculate the cosine similarity between the query embedding and the combined embeddings
    similarity_scores = cosine_similarity_faiss(
        query_embedding, combined_embeddings
    ).flatten()

    # Get the indices of the top n most relevant strings
    top_indices = np.argsort(similarity_scores)[-n:][::-1]

    # Get the top n most relevant strings
    top_strings = filtered_df.loc[top_indices, "combined"].tolist()

    return top_strings


def get_relevant_data(query, df, code_to_desc, n):
    """
    Get the relevant data for a given question.

    Args:
        question (str): The question to get the relevant data for.
        df (pandas.DataFrame): The dataframe containing the strings to search in.
        nq (int): The number of top relevant strings to return if the query is a code.
        np (int): The number of top relevant strings to return if the query is the question.

    Returns:
        list: A list of the relevant data for the question.
    """

    relevant_data = []

    if code_to_desc is not None:
        codes_to_query = extract_codes(query)
        for code in codes_to_query:
            if code in code_to_desc:
                relevant_data.append(code_to_desc[code])

    if df is not None:
        if len(relevant_data) < n:
            n = n - len(relevant_data)
        else:
            n = 1
        filtered_df = deepcopy(df[~(df["combined"].isin(relevant_data))].reset_index())
        relevant_data_from_similarity_search = search_relevant_strings(
            query, filtered_df, n
        )
        # print(relevant_data_from_similarity_search)
        relevant_data = relevant_data + relevant_data_from_similarity_search
    # print(len(relevant_data), type(relevant_data))

    return "\n\n".join(set(relevant_data))


def get_messages(
    user_prompt_template,
    question,
    n,
    system_prompt=None,  # "You are a expert in Medical Coding (ICD10, ICD9, CPT Codes, etc.)",
    df_for_rag=None,  # Dataframe with embeddings
    code_to_desc=None,  # code to description mapping
):
    if "option_a" in question:
        question["options"] = (
            f'a. {question["option_a"]} b. {question["option_b"]} c. {question["option_c"]} d. {question["option_d"]}'
        )
        question_string = f"{question['question']}\n{question['options']}"
    else:
        question_string = f"{question['question']}"

    format_args = {"question_string": question_string}
    if "note" in question:
        format_args["note"] = question["note"]

    if "note" in question:
        query = f"{question['note']}\n\nQuestion: {question_string}"
    else:
        query = question_string

    relevant_data_string = get_relevant_data(
        query, df=df_for_rag, code_to_desc=code_to_desc, n=n
    )

    if relevant_data_string == "":
        user_prompt_template = user_prompt_template.replace(
            "\n\nRelevant Data: ```{relevant_data}```", ""
        )
    else:
        format_args["relevant_data"] = relevant_data_string

    user_prompt = user_prompt_template.format(**format_args)

    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt},
        ]

    return messages


async def generate_answer(messages, temperature, top_p):
    """
    Send a prompt to OpenAI API and get the answer.
    """
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    completion = await client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_ID"),
        messages=messages,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content


async def async_run_questions_from_json(
    test_questions: dict,
    results_file: str,
    results_dir: str,
    user_prompt_template: str = None,
    user_prompt_template_with_note: str = None,
    df_for_rag: pd.DataFrame = None,
    code_to_desc: dict = None,
    temperature: float = None,
    top_p: float = None,
    system_prompt: str = None,
    n: int = 10,
):
    questions = []
    answer_letters = []
    categories = []
    explanations = []
    model_answers = []
    notes = []
    messages_list = []
    prompts = []
    tasks = []
    print(
        "Adding prompts for execution. Please wait until the results are accumulated and stored to file before running evaluation."
    )
    for _, question in tqdm(enumerate(test_questions), total=len(test_questions)):
        # prepare messages for the question

        if "note" in question:
            if user_prompt_template_with_note is None:
                raise ("user_prompt_template_with_note expected but None.")
            prompt_template = user_prompt_template_with_note
        else:
            prompt_template = user_prompt_template

        messages = get_messages(
            user_prompt_template=prompt_template,
            question=question,
            system_prompt=system_prompt,
            code_to_desc=code_to_desc,
            df_for_rag=df_for_rag,
            n=n,
        )
        if len(messages) == 2:
            prompts.append(messages[1]["content"])
        else:
            prompts.append(messages[0]["content"])
        questions.append(question["question"])
        answer_letters.append(question["answer_letter"])
        if "note" in question:
            notes.append(question["note"])

        categories.append(
            question["subject"] if "subject" in question else question["category"]
        )
        messages_list.append(messages)
        explanations.append(question["explanation"])

        task = asyncio.create_task(generate_answer(messages, temperature, top_p))
        tasks.append(task)

    model_answers = await asyncio.gather(*tasks)

    # create a dictionary with the question and model answer
    results = {
        "question": questions,
        "answer_letter": answer_letters,
        "category": categories,
        "explanation": explanations,
        "model_answer": model_answers,
        "note": notes if len(notes) == len(questions) else [None] * len(questions),
        "messsages": messages_list,
        "prompt": prompts,
    }

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_dir, results_file)
    results_df.to_csv(results_file, index=False)

# ------------------------------
# Functions below are from the openai-cookbook repository
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_combine_GPT4v_with_RAG_Outfit_Assistant.ipynb
# ------------------------------
# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(10))
def get_embeddings(input: List) -> List:
    """
    Retrieves embeddings for the given input using the specified embedding model.

    Args:
        input (List): A list of input data.

    Returns:
        List: A list of embeddings corresponding to the input data.

    Raises:
        RuntimeError: If an error occurs while retrieving embeddings from OpenAI.

    Examples:
        >>> input_data = ["Hello", "World"]
        >>> embeddings = get_embeddings(input_data)
        >>> print(embeddings)
        [0.123, 0.456]
    """
    # try:
    if True:
        response = client.embeddings.create(
            input=input, model=os.getenv("EMBEDDING_MODEL")
        ).data
        return [data.embedding for data in response]
    # except Exception as e:
    #     raise RuntimeError(
    #         "Error occurred while retrieving embeddings from OpenAI"
    #     ) from e


def batchify(iterable, n=1):
    """
    Batchify an iterable into smaller chunks of size n.

    Args:
        iterable: The iterable to be batchified.
        n (int): The size of each batch. Defaults to 1.

    Yields:
        list: A batch of elements from the iterable.

    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def embed_corpus(
    corpus: List[str],
    batch_size=64,
    num_workers=8,
    max_context_len=8191,
):
    """
    Embeds a corpus of text using a pre-trained model.

    Args:
        corpus (List[str]): The list of text articles to embed.
        batch_size (int, optional): The batch size for embedding. Defaults to 64.
        num_workers (int, optional): The number of worker threads for parallel embedding. Defaults to 8.
        max_context_len (int, optional): The maximum length of each article to consider. Defaults to 8191.

    Returns:
        List[Embedding]: The list of embeddings for each article in the corpus.
    """

    # Encode the corpus, truncating to max_context_len
    encoding = tiktoken.get_encoding("cl100k_base")
    encoded_corpus = [
        encoded_article[:max_context_len]
        for encoded_article in encoding.encode_batch(corpus)
    ]

    # Calculate corpus statistics: the number of inputs, the total number of tokens, and the estimated cost to embed
    num_tokens = sum(len(article) for article in encoded_corpus)
    cost_to_embed_tokens = num_tokens / 1000 * EMBEDDING_COST_PER_1K_TOKENS
    print(
        f"num_articles={len(encoded_corpus)}, num_tokens={num_tokens}, est_embedding_cost={cost_to_embed_tokens:.2f} USD"
    )

    # Embed the corpus
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [
            executor.submit(get_embeddings, text_batch)
            for text_batch in batchify(encoded_corpus, batch_size)
        ]

        with tqdm(total=len(encoded_corpus)) as pbar:
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(batch_size)

        embeddings = []
        for future in futures:
            data = future.result()
            embeddings.extend(data)

        return embeddings


# Function to generate embeddings for a given column in a DataFrame
def generate_embeddings(df, column_name):
    """
    Generate embeddings for the given column in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to generate embeddings for.

    Returns:
        None

    Raises:
        None

    """
    # Initialize an empty list to store embeddings
    descriptions = df[column_name].astype(str).tolist()
    embeddings = embed_corpus(descriptions)

    # Add the embeddings as a new column to the DataFrame
    df["embeddings"] = embeddings
    print("Embeddings created successfully.")