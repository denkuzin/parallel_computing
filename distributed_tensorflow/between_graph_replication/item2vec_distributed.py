import tensorflow as tf
import socket
import time
import os
import multiprocessing
import logging


logger = logging.getLogger()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NUM_WORKERS = 10
WORKER_CPU_NUM = multiprocessing.cpu_count()
PREFIX = "tf"
BATCH_SIZE = 64
embedding_size = 400
learning_rate = 1.0


NUM_EPOCHS = 1
MAX_STEPS = int(1e10)
right_number = 48607492 
left_number = 184149040


INPUT_PATTERN = 'gs://my_bucket/experiments/data/part*'
results_path = 'gs://my_bucket/experiments/results'
board_path = 'gs://my_bucket/experiments/board'


def variable_summaries(var, scope_name='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(scope_name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean_{}'.format(scope_name), mean)
        tf.summary.histogram('histogram_{}'.format(scope_name), var)


class Cluster:
    """to describe claster;
       based on standart Compute Cloud Machine names
    """
    def get_job_name(self):
        if self.hostname == '{}-m'.format(self.prefix):
            return "ps"  #"client"
        elif self.hostname == '{}-w-0'.format(self.prefix):
            return "worker"
        else:
            return "worker"

    def get_task_index(self):
        if self.hostname == '{}-m'.format(self.prefix):
            return 0
        else:
            return int(self.hostname.split("-")[-1])

    def get_clust_spec(self):
        return {
              "ps": ['{}-m:{}'.format(self.prefix, self.port)],
              "worker": [ '{}-w-{}:{}'.format(self.prefix,i,self.port) \
                          for i in range(self.num_workers)]}

    def __init__(self, prefix, num_workers):
        self.prefix = prefix
        self.num_workers = num_workers
        self.hostname = socket.gethostname()
        self.port = "2222"
        self.job_name = self.get_job_name()
        self.task_index = self.get_task_index()
        self.clust_spec = self.get_clust_spec()


def input_pipeline(read_threads=8,
                   input_pattern=None,
                   batch_size=64,
                   num_epochs=None,
                   task_index=None,
                   num_workers=None):

    enqueue_many_size = batch_size
    min_after_dequeue = 5 * batch_size
    capacity = min_after_dequeue + (read_threads + 1) * batch_size

    files_names = tf.train.match_filenames_once(input_pattern)
    to_process = tf.strided_slice(files_names, [task_index],
                                  [999999999], strides=[num_workers])
    filename_queue = tf.train.string_input_producer(to_process, capacity=32*2,
                                          shuffle=True, num_epochs=num_epochs)
    reader = tf.TextLineReader()
    _, queue_batch = reader.read_up_to(filename_queue, enqueue_many_size)

    batch_serialized_example = tf.train.shuffle_batch(
        [queue_batch],
        batch_size=batch_size,
        num_threads=read_threads,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True)

    record_defaults = [tf.constant([], dtype=tf.int32),
                       tf.constant([], dtype=tf.int32)]
    decoded = tf.decode_csv(records=batch_serialized_example,
                            record_defaults=record_defaults,
                            field_delim='\t')
    decoded[1] = tf.reshape(decoded[1], shape=(batch_size, 1))
    return decoded


def main(_):
    #get info about cluster

    MyClaster = Cluster(PREFIX, NUM_WORKERS)
    cluster = tf.train.ClusterSpec(MyClaster.clust_spec)
    server = tf.train.Server(cluster,
                           job_name=MyClaster.job_name,
                           task_index=MyClaster.task_index)

    # for parameter servers, wait for incoming connections forever
    if MyClaster.job_name == "ps":
        server.join()

    # for workers
    elif MyClaster.job_name == "worker":

        # Build and train a skip-gram model
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % MyClaster.task_index, cluster=cluster)):

            with tf.name_scope('input'):
                decoded = input_pipeline(
                            read_threads=WORKER_CPU_NUM,
                            input_pattern=INPUT_PATTERN,
                            batch_size=BATCH_SIZE,
                            num_epochs=NUM_EPOCHS,
                            task_index=MyClaster.task_index,
                            num_workers=MyClaster.num_workers)
                train_inputs, train_labels = decoded

            with tf.device("/job:ps/task:0"):
                embeddings_1 = tf.Variable(
                      tf.random_uniform([left_number, embedding_size], -0.05, 0.05),
                      name="embeddings_1")
                embeddings_2 = tf.Variable(
                          tf.random_uniform([right_number, embedding_size], -0.05, 0.05),
                          name="embeddings_2")
                biases = tf.Variable(tf.zeros([right_number]), name="biases")

            embed = tf.nn.embedding_lookup(embeddings_1, train_inputs, name="embed")

            with tf.name_scope('LOSS'):
                loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                           weights=embeddings_2,
                           biases=biases,
                           inputs=embed,
                           labels=train_labels,
                           num_sampled=7,
                           num_classes=right_number,
                           remove_accidental_hits=True))

            with tf.name_scope('train'):
                global_step = tf.contrib.framework.get_or_create_global_step()
                optimizer = tf.train.GradientDescentOptimizer(learning_rate, use_locking=False).\
                                  minimize(loss, global_step=global_step)

            with tf.name_scope('norms'):
                with tf.device("/job:ps/task:0"):
                    norm_left = tf.sqrt(tf.reduce_sum(tf.square(embeddings_1), 1, keep_dims=True), name='norm_left')
                    norm_right = tf.sqrt(tf.reduce_sum(tf.square(embeddings_2), 1, keep_dims=True), name='norm_right')
                    if MyClaster.task_index in (2, 3):
                        variable_summaries(norm_left, scope_name='norm_left')
                        variable_summaries(norm_right, scope_name='norm_right')

        # The StopAtStepHook handles stopping after running given steps
        hooks = [tf.train.StopAtStepHook(last_step=MAX_STEPS)]

        # Each worker only needs to contact the PS task(s) and the local worker task:       
        config = tf.ConfigProto(
                       device_filters=['/job:ps', '/job:worker/task:%d' % MyClaster.task_index],
                       intra_op_parallelism_threads=0,
                       inter_op_parallelism_threads=0,
                       operation_timeout_in_ms=60000*10
        )

        merged = tf.summary.merge_all()

        with tf.train.MonitoredTrainingSession(
            config=config,
            master=server.target,
            is_chief=(MyClaster.task_index == 0 and (MyClaster.job_name == 'worker')),
            checkpoint_dir=results_path,
            save_checkpoint_secs=3600,
            save_summaries_steps=20000000//BATCH_SIZE,
            hooks=hooks,
            log_step_count_steps=20000000//BATCH_SIZE) as mon_sess:

            writer = tf.summary.FileWriter(board_path, mon_sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=mon_sess, coord=coord)

            with coord.stop_on_exception():

                with tf.control_dependencies(threads):
                    logger.info(str(threads))
                average_loss = 0
                counter = 1
                ts = time.time()

                while not coord.should_stop() and not mon_sess.should_stop():
                    num_lines = 1000000
                    if counter*BATCH_SIZE % (num_lines*100) == 0 and MyClaster.task_index == 1:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, loss_val = mon_sess.run(
                            [optimizer, loss], options=run_options, run_metadata=run_metadata)
                        average_loss += loss_val
                        writer.add_run_metadata(run_metadata, 'step%d' % counter)
                        logger.info("task_ind = {}, metadata and summary are saved".format(MyClaster.task_index))
                    else:
                        if counter*BATCH_SIZE % num_lines == 0:
                            _, loss_val, summary = mon_sess.run([optimizer, loss, merged])
                            average_loss += loss_val
                            writer.add_summary(summary, counter)
                        else:
                            _, loss_val = mon_sess.run([optimizer, loss])
                            average_loss += loss_val

                    #print speed
                    if counter*BATCH_SIZE % num_lines == 0:
                        speed = num_lines/(time.time() - ts)
                        logger.info("task_ind = {}, loss = {}, counter = {}, speed = {:.1f} l/s".
                                       format(MyClaster.task_index,
                                            average_loss/(num_lines/BATCH_SIZE), counter, speed))

                        summary_loss_average = tf.Summary()
                        summary_loss_average.value.add(tag="loss", simple_value=average_loss/(num_lines/BATCH_SIZE))
                        summary_loss_average.value.add(tag="speed", simple_value=speed)
                        writer.add_summary(summary_loss_average, counter)

                        ts = time.time()
                        average_loss = 0
                    counter += 1

            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    tf.app.run(main=main)
