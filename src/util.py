from functools import wraps
from time import time
import shutil
import re
from os import cpu_count
from multiprocessing import Queue, Process
import networkx as nx
from tqdm import tqdm
from os.path import join


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


def clean(dataset_name="enron", output_folder="output/"):
    path = join(output_folder, dataset_name)
    shutil.rmtree(path, ignore_errors=True)


def parallel_for(f, params0, params1, prange, cores=None, yielding=True, DEBUG=False):
    n = len(prange)
    workers = cpu_count() if cores is None else cores
    if DEBUG:
        pb = tqdm(total=n)
        pb.refresh()
    queue = Queue()
    it = iter(prange)
    procs = []
    res = {}
    started, ended = 0, 0
    while n > ended:
        while n > started and len(procs) - ended < workers:
            started += 1
            j = next(it)
            p = Process(target=f, args=(queue, j, params0[j], params1))
            p.start()
            procs += [p]
        while not queue.empty():
            j, resj = queue.get()
            if yielding:
                yield resj
            res[j] = resj
            ended += 1
            if DEBUG:
                pb.update(1)
    next(it, None)
    for p in procs:
        p.join()
        p.close()
    if not yielding:
        return res


def parallel_for_balanced(f, params, prange, cores=None, DEBUG=False):
    n = len(prange)
    workers = cpu_count() if cores is None else cores
    if DEBUG:
        pb = tqdm(total=n)
        pb.refresh()
    its = [[] for _ in range(workers)]
    for i, it in enumerate(prange):
        its[i % workers].append(it)
    res = {}
    procs = []
    queue = Queue()
    for i in range(workers):
        p = Process(target=f, args=(queue, its[i], *params))
        p.start()
        procs += [p]
    ended = 0
    while ended < workers:
        while not queue.empty():
            c = queue.get()
            if c is not None:
                j, resj = c
                res[j] = resj
                if DEBUG:
                    pb.update(1)
            else:
                ended += 1
    for p in procs:
        p.join()
        p.close()
    return res


def worker_average_shortest_path_length(queue, iis, nodes, g):
    for i in iis:
        sp = nx.single_source_shortest_path_length(g, nodes[i]).values()
        dst = sum(sp)
        queue.put((i, dst))
    queue.put(None)


def par_average_shortest_path_length(g):
    n = g.number_of_nodes()
    print("Computing par_average_shortest_path_length")
    if n >= 10**5:
        return -1
    res = parallel_for_balanced(
        worker_average_shortest_path_length, (list(g.nodes()), g), range(n), DEBUG=True
    ).values()
    ans = sum(res) / (n * (n - 1))
    return ans


@timing
def calculate_avg_path_length(UG):
    try:
        if nx.is_connected(UG):
            # return nx.average_shortest_path_length(UG)
            return par_average_shortest_path_length(UG)
        else:
            largest_cc = max(nx.connected_components(UG), key=len)
            subgraph = UG.subgraph(largest_cc)

            # TOCHECK: might be too slow for big networks
            # return nx.average_shortest_path_length(subgraph)
            return par_average_shortest_path_length(subgraph)

    except Exception as e:
        print(f"Error calculate_avg_path_length: {e}")
        exit(0)


sanitized_set = {}


def sanitize_filename(name):
    sname = ""
    for c in name:
        if re.search(r"[a-zA-Z0-9_\-]", c) is None:
            if c not in sanitized_set:
                nc = str(len(sanitized_set))
                sanitized_set[c] = nc
                c = nc
            else:
                c = sanitized_set[c]
        sname += c
    return sname
