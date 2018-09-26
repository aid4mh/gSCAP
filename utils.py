#!/usr/bin/env python

""" A collection of data processing scripts """

from multiprocessing import Manager, Queue, Process


__author__ = 'Luke Waninger'
__copyright__ = 'Copyright 2018, University of Washington'
__credits__ = 'Abhishek Pratap'

__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Luke Waninger'
__email__ = 'luke.waninger@gmail.com'
__status__ = 'development'


def progress_bar(pbar):
    """progress bar to track parallel events
    Args:
        total: (int) total number of tasks to complete
        desc: (str) optional title to progress bar
    Returns:
        (Process, Queue)
    """
    proc_manager = Manager()

    def track_it(pbar, trackq):
        idx = 0

        while True:
            try:
                update = trackq.get()

                if update is None:
                    break

            except EOFError:

                break

            pbar.update(update)
            idx += 1

    trackq = proc_manager.Queue()
    p = Process(target=track_it, args=(pbar, trackq))
    p.start()

    return p, trackq


def pcmap(func, vals, n_cons, pbar=False, **kwargs):
    """parallel mapping function into producer-consumer pattern
    Args:
        func: (Function) function to apply
        vals: [Object] values to apply
        n_cons: (int) number of consumers to start
        pbar: (bool)

    Returns:
        [Object] list of mapped function return values
    """
    if pbar:
        total = len(vals)
        desc = kwargs.get('desc')

        pbar, trac_qu = progress_bar(total, desc)
    else:
        pbar, trac_qu = None, None

    def consumer(c_qu, r_qu, func):
        """consumer, terminate on receiving 'END' flag
        Args:
            c_qu: (Queue) consumption queue
            r_qu: (Queue) results queue
        """
        while True:
            val = c_qu.get()

            if isinstance(val, str) and val == 'END':
                break

            rv = func(val)
            r_qu.put(rv)

        r_qu.put('END')

    # create queues to pass tasks and results
    consumption_queue = Queue()
    results_queue = Queue()

    # setup the consumers
    consumers = [
        Process(target=consumer, args=(
            consumption_queue,
            results_queue,
            func
        ))
        for i in range(n_cons)
    ]

    # start the consumption processes
    [c.start() for c in consumers]

    # dish out tasks and add the termination flag
    [consumption_queue.put(val) for val in vals]
    [consumption_queue.put('END') for c in consumers]

    # turn the results into a list
    running, brake, results = n_cons, False, []
    while not brake:
        while not results_queue.empty():
            val = results_queue.get()

            if isinstance(val, str) and val == 'END':
                running -= 1

                if running == 0:
                    brake = True
            else:
                if trac_qu is not None:
                    trac_qu.put(1)
                else:
                    pass

                results.append(val)

    # kill and delete all consumers
    [c.terminate() for c in consumers]
    del consumers

    # kill the progress bar
    if pbar is not None:
        pbar.terminate()
        pbar.join()
        del pbar, trac_qu
    else:
        pass

    return results


if __name__ == '__main__':
    pass
