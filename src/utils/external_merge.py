import logging
import multiprocessing
import os
import queue
import shutil
import sys
import time

import numpy as np

from utils.multiprocess_queue import MultiprocessQueue, SharedCounter


def numpy_load_full_file_to_array(file_name, data_type, num_component):
    full_bytes = os.path.getsize(file_name)
    row_size = int(full_bytes / (data_type.itemsize * num_component))
    file_reader = np.memmap(file_name, dtype=data_type, mode='r',
                            shape=(row_size, num_component))
    return np.array(file_reader)


def numpy_file_reader_generating_function(file_name, data_type, batch_size, num_component):
    remain_bytes = os.path.getsize(file_name)
    batch_bytes = data_type.itemsize * batch_size * num_component
    offset = 0

    assert isinstance(data_type, np.dtype)
    assert remain_bytes % (data_type.itemsize * num_component) == 0

    while remain_bytes:
        if remain_bytes < batch_bytes:
            # Construct the last inner iterator
            last_items_size = int(remain_bytes / (data_type.itemsize * num_component))
            file_reader = np.array(
                np.memmap(file_name, dtype=data_type, mode='r', shape=(last_items_size, num_component), offset=offset)
            )
            # Update counter
            offset += remain_bytes
            remain_bytes = 0

            # generate next output
            for i in range(last_items_size):
                yield file_reader[i]
        else:
            # Construct inner iterator
            file_reader = np.array(
                np.memmap(file_name, dtype=data_type, mode='r', shape=(batch_size, num_component), offset=offset)
            )
            # Update counter
            offset += batch_bytes
            remain_bytes -= batch_bytes

            # generate next output
            for i in range(batch_size):
                yield file_reader[i]


def numpy_file_merger_generating_function(file1_name, file2_name, data_type, batch_size, num_component, less_than_func):
    file1_reader = numpy_file_reader_generating_function(file1_name, data_type, batch_size, num_component)
    file2_reader = numpy_file_reader_generating_function(file2_name, data_type, batch_size, num_component)

    row1 = next(file1_reader)
    row2 = next(file2_reader)

    while True:
        if less_than_func(row1[:num_component - 1], row2[:num_component - 1]):
            yield row1
            try:
                row1 = next(file1_reader)
            except StopIteration:
                yield row2
                while True:
                    yield next(file2_reader)
        else:
            yield row2
            try:
                row2 = next(file2_reader)
            except StopIteration:
                yield row1
                while True:
                    yield next(file1_reader)


def numpy_merge_files(new_file_name, file1_name, file2_name, data_type, batch_size, num_component, less_than_func):
    """
    Merge file1_name, file2_name into new_file_name in sorted manner,
    the inplace buffer size is batch_size*num_component*data_type size
    and the sorting is based on the comparison less_than_func
    """
    full_bytes = os.path.getsize(file1_name) + os.path.getsize(file2_name)
    row_size = int(full_bytes / (data_type.itemsize * num_component))

    if os.path.exists(new_file_name):
        shutil.rmtree(new_file_name)
    file_writer = np.memmap(new_file_name, dtype=data_type, mode='w+', shape=(row_size, num_component))

    file_reader = numpy_file_merger_generating_function(file1_name=file1_name, file2_name=file2_name,
                                                        data_type=data_type,
                                                        batch_size=batch_size,
                                                        num_component=num_component,
                                                        less_than_func=less_than_func
                                                        )
    buffer = np.zeros((batch_size, num_component), dtype=data_type)
    remaining_size = row_size
    offset = 0
    while remaining_size:
        if remaining_size < batch_size:
            for i in range(remaining_size):
                buffer[i] = next(file_reader)
            file_writer[offset:offset+remaining_size, :] = buffer[:remaining_size]

            remaining_size = 0
            offset = row_size
        else:
            for i in range(batch_size):
                buffer[i] = next(file_reader)
            file_writer[offset:offset+batch_size, :] = buffer[:]

            remaining_size -= batch_size
            offset += batch_size

    file_writer.flush()
    return new_file_name


class Merger(multiprocessing.Process):
    """External numpy records sort worker"""
    QUEUE_TIMEOUT = 0.5

    def __init__(
            self, sorted_slice_queue, directory_name,
            queue_lock, poison_pill, job_counter, max_job_count, total_worker_counter,
            data_type, batch_size, num_component, less_than_func
    ):
        # sorted_slice_queue should be a MultiprocessQueue object
        self.sorted_slice_queue = sorted_slice_queue
        self.dir = directory_name

        self.queue_lock = queue_lock
        self.poison_pill = poison_pill
        # shared_counter should be a SharedCounter object
        self.shared_counter = job_counter
        self.max_job_count = max_job_count
        self.total_worker_counter = total_worker_counter
        self.total_buffer = total_worker_counter.value * batch_size

        self.data_type = data_type
        self.batch_size = batch_size
        self.num_component = num_component
        self.less_than_func = less_than_func
        super(Merger, self).__init__()

    def run(self):
        try:
            while not self.poison_pill.is_set() and self.shared_counter.value < self.max_job_count:
                if self.sorted_slice_queue.empty():     # workers number is bigger the left job
                    self.total_worker_counter.decrement()
                    break
                with self.queue_lock:   # Atomically take two elements from queue
                    try:
                        file_name_1 = self.sorted_slice_queue.get(timeout=self.QUEUE_TIMEOUT)
                    except queue.Empty:
                        continue
                    try:
                        file_name_2 = self.sorted_slice_queue.get(timeout=self.QUEUE_TIMEOUT)
                    except queue.Empty:
                        self.sorted_slice_queue.put(file_name_1)  # Just put it back
                        continue

                self.shared_counter.increment()

                tic = time.perf_counter()
                # dynamically get larger buffer
                self.batch_size = int(self.total_buffer/self.total_worker_counter.value)
                merged_file_name = numpy_merge_files(
                    new_file_name=os.path.join(self.dir, str(self.shared_counter.value)),
                    file1_name=file_name_1, file2_name=file_name_2,
                    data_type=self.data_type,
                    batch_size=self.batch_size,
                    num_component=self.num_component,
                    less_than_func=self.less_than_func
                )
                toc = time.perf_counter()
                os.unlink(file_name_1)
                os.unlink(file_name_2)
                self.sorted_slice_queue.put(merged_file_name)
                logging.info(f"Merged sorted slice {file_name_1} and {file_name_2} to {merged_file_name} "
                             f"in {toc - tic:0.4f} seconds")

        except Exception as ex:
            logging.exception(ex)
            self.poison_pill.set()


def parallel_external_merger(
        sorted_slice_list, directory_name,
        data_type, batch_size, num_component, less_than_func,
        outfile_name,
        num_workers
):
    # Construct the workers for merge job
    sorted_slice_queue = MultiprocessQueue()
    job_counter = SharedCounter()
    total_worker_counter = SharedCounter()
    total_worker_counter.increment(num_workers)
    queue_lock = multiprocessing.Lock()
    for file in sorted_slice_list:
        sorted_slice_queue.put(file)
    poison_pill = multiprocessing.Event()

    pool = [
        Merger(
            sorted_slice_queue=sorted_slice_queue, directory_name=directory_name,
            queue_lock=queue_lock, poison_pill=poison_pill, job_counter=job_counter,
            total_worker_counter=total_worker_counter,
            max_job_count=len(sorted_slice_list) - 1,
            data_type=data_type, batch_size=batch_size,
            num_component=num_component, less_than_func=less_than_func
        ) for _ in range(num_workers)
    ]

    # workers began to work
    for p in pool:
        p.start()

    while len(pool):
        for p in pool:
            if not p.is_alive():
                if p.exitcode:
                    poison_pill.set()
                    logging.error(f"merger {p} exits with {p.exitcode}")
                else:
                    logging.info(f"merger {p} successfully exits")
                pool.remove(p)
        time.sleep(1)

    if not poison_pill.is_set():
        merged_file_name = sorted_slice_queue.get()
        shutil.move(merged_file_name, outfile_name)
        return outfile_name
    else:
        sys.exit('Fail to complete the external merge job, program terminates...')
