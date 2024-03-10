import multiprocessing
from multiprocessing.pool import ThreadPool
import json
import pdb
from tqdm import tqdm
import jnius_config

import bm25_utils as utils
from base_retriever import Retriever


def _run_thread(arguments):
    idz = arguments["id"]
    index = arguments["index"]
    k = arguments["k"]
    data = arguments["data"]

    # BM25 parameters #TODO
    # bm25_a = arguments["bm25_a"]
    # bm25_b = arguments["bm25_b"]
    # searcher.set_bm25(bm25_a, bm25_b)
    from pyserini.search.lucene import LuceneSearcher
    # from pyserini.search import SimpleSearcher
    # pdb.set_trace()
    searcher = LuceneSearcher(index)

    _iter = data
    if idz == 0:
        _iter = tqdm(data)

    provenance = {}
    for x in _iter:
        query_id = x["id"]
        query = (
            x["query"].replace(utils.ENT_END, "").replace(utils.ENT_START, "").strip()
        )

        hits = searcher.search(query, k)

        element = []
        for y in hits:
            # pdb.set_trace()
            try:
                doc_data = json.loads(str(y.docid).strip())
                doc_data["score"] = y.score
                doc_data["text"] = y.lucene_document.get('contents').strip()
                element.append(doc_data)
            except Exception as e:
                # print(e)
                page_id, par_id = y.docid.split('_')
                # pdb.set_trace()
                element.append(
                    {
                        "score": y.score,
                        "text": y.lucene_document.get('contents').strip(),
                        "page_id": page_id,
                        "start_par_id": (int)(par_id),
                        "end_par_id" : (int)(par_id),
                    }
                )
        provenance[query_id] = element

    return provenance


class BM25(Retriever):
    def __init__(self, name, index, k, num_threads, Xms=None, Xmx=None):
        super().__init__(name)
        # pdb.set_trace()
        if Xms and Xmx:
            # to solve Insufficient memory for the Java Runtime Environment
            jnius_config.add_options(
                "-Xms{}".format(Xms), "-Xmx{}".format(Xmx), "-XX:-UseGCOverheadLimit"
            )
            print("Configured options:", jnius_config.get_options())

        self.num_threads = min(num_threads, int(multiprocessing.cpu_count()))

        # initialize a ranker per thread
        self.arguments = []
        for id in tqdm(range(self.num_threads)):
            self.arguments.append(
                {
                    "id": id,
                    "index": index,
                    "k": k,
                }
            )

    def feed_data(self, queries_data, logger=None):

        chunked_queries = utils.chunk_it(queries_data, self.num_threads)

        for idx, arg in enumerate(self.arguments):
            arg["data"] = chunked_queries[idx]

    def run(self):
        pool = ThreadPool(self.num_threads)
        results = pool.map(_run_thread, self.arguments)

        provenance = {}
        for x in results:
            provenance.update(x)
        pool.terminate()
        pool.join()

        return provenance
