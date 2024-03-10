import json
import argparse

import retrieval
import bm25_utils as utils

def execute(
    logger, test_config_json, retriever, log_directory, model_name, output_folder
):
    # run evaluation
    retrieval.run(
        test_config_json, retriever, model_name, logger, output_folder=output_folder
    )

def main(args):

    # load configs
    with open(args.test_config, "r") as fin:
        test_config_json = json.load(fin)
    # create a new directory to log and store results
    log_directory = utils.create_logdir_with_timestamp(args.logdir)
    logger = None

    logger = utils.init_logging(log_directory, 'bm25', logger)
    logger.info("loading {} ...".format('bm25'))

    import BM25_connector

    if args.model_configuration:
        retriever = BM25_connector.BM25.from_config_file(
            'bm25', args.model_configuration
        )
    else:
        retriever = BM25_connector.BM25.from_default_config('bm25')

    execute(
        logger,
        test_config_json,
        retriever,
        log_directory,
        'bm25',
        args.output_folder,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_config",
        dest="test_config",
        type=str,
        help="Test Configuration.",
    )

    parser.add_argument(
        "--logdir",
        dest="logdir",
        type=str,
        default="logs/ranking/",
        help="logdir",
    )

    parser.add_argument(
        "--model_configuration",
        "-c",
        dest="model_configuration",
        type=str,
        default=None,
        help="model configuration",
    )

    parser.add_argument(
        "--output_folder",
        "-o",
        dest="output_folder",
        type=str,
        required=True,
        help="output folder",
    )

    args = parser.parse_args()

    main(args)
