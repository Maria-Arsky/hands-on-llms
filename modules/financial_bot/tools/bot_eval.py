import logging
import json
import statistics
import fire

from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query, context, response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    from ragas.metrics import context_precision, context_recall, answer_similarity, faithfulness
    metrics = [context_precision, context_recall, answer_similarity, faithfulness]
    results = {"context_precision": [], "context_recall": [], "answer_similarity": [], "faithfulness": []}

    with open(testset_path, "r") as f:
        data = json.load(f)
        for idx, elem in enumerate(data):
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload).split('\n')
            input_payload['context'] = output_context
            output_reasoning = bot.finbot_chain.chains[1].run(**input_payload).split('\n')
            del input_payload['context']
            response = bot.answer(**input_payload)
            logger.info("ABOUT = %s", elem["about_me"])
            logger.info("QUESTION = %s", elem["question"])
            logger.info("REASONING = %s", output_reasoning)
            logger.info("CONTEXT = %s", output_context)
            logger.info("REF_RESPONSE = %s", elem["response"])
            logger.info("RESPONSE = %s", response)
            score = evaluate_w_ragas(query=elem["question"], context=output_context, output=response, ground_truth=elem["response"], metrics=metrics)
            logger.info("SCORE=%s", score)
            
            for metric in results.keys():
                results[metric].append(score[metric])
        
    avg_str, med_str = "", ""
    for metric in results.keys():
        avg = statistics.mean(results[metric])
        avg_str += f"{metric}: {avg}, "
        med = statistics.median(results[metric])
        med_str += f"{metric}: {med}, "
        
    logger.info(f"AVERAGE_SCORE = {avg_str}")
    logger.info(f"MEDIAN_SCORE = {med_str}")

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
