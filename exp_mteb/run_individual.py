import mteb
from sentence_transformers import SentenceTransformer
import argparse

import os
os.environ['HF_HOME'] = '/data1/lsj9862/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data1/lsj9862/huggingface/datasets'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="BAAI/bge-base-en-v1.5") # ["BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2", "thenlper/gte-base"]
    parser.add_argument("--task", type=str, default="all", choices=["retrieval", "classification", "sts", "all"])
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)

    TASK = []

    if args.task in ["retrieval", "all"]:
        RETRIEVAL_TASKS = [
            "SCIDOCS",
            "LegalBenchCorporateLobbying",
            "BelebeleRetrieval",
            "WikipediaRetrievalMultilingual",
            "StackOverflowQA"
            ]
        TASK.extend(RETRIEVAL_TASKS)

    elif args.task in ["classification", "all"]:
        CLASSIFICATION_TASKS = [
            "FinancialPhrasebankClassification",
            "SwissJudgementClassification",
            "PoemSentimentClassification",
            "MassiveIntentClassification",
            "TweetTopicSingleClassification"
        ]
        TASK.extend(CLASSIFICATION_TASKS)

    elif args.task in ["sts", "all"]:
        STS_TASKS = [
            "STSBenchmark",
            "FinParaSTS",
            "SICK-R",
            "STS22.v2",
            "SemRel24STS",
            "STS12",
            "STS13",
            "STS14",
            "STS15",
            "STS17"
        ]
        TASK.extend(STS_TASKS)


    # Specify the model that we want to evaluate
    model = SentenceTransformer(args.model_name)

    # specify what you want to evaluate it on
    tasks = mteb.get_tasks(tasks=TASK)

    # run the evaluation
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.run(
            model,
        output_folder=f"{args.output_folder}/{args.model_name}",
        batch_size=args.batch_size
    )
    print(f"Results saved to {args.output_folder}/{args.model_name}")