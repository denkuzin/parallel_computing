import tensorflow as tf
import socket


PREFIX = "tf"

NUM_WORKERS = 3
BATCH_SIZE = 100
embedding_size = 400
learning_rate = 1.0

NUM_EPOCHS = 2
MAX_STEPS = 10000

INPUT_PATTERN = "gs://my_bucket/experiments/test-data/data*txt"
path_left = "experiments/test-data/reverse_leftdict.pickle"
path_right = "experiments/test-data/reverse_rightdict.pickle"


class Cluster:
    def __init__(self, prefix, num_workers):
        self.prefix = prefix
        self.num_workers = num_workers
        self.hostname = socket.gethostname()
        self.port = "2222"
        self.job_name = self.get_job_name()
        self.task_index = self.get_task_index()
        self.clust_spec = self.get_clust_spec()

    def get_job_name(self):
        if self.hostname == '{}-m'.format(self.prefix):
            return "client"
        elif self.hostname == '{}-w-0'.format(self.prefix):
            return "ps"
        else:
            return "worker"

    def get_task_index(self):
        if self.hostname == '{}-m'.format(self.prefix):
            return 0
        else:
            return int(self.hostname.split("-")[-1])

    def get_clust_spec(self):
        return {
                "client": ['{}-m:{}'.format(self.prefix, self.port)],
                "ps": ['{}-w-0:{}'.format(self.prefix, self.port)],
                "worker": ['{}-w-{}:{}'.format(self.prefix, i, self.port)
                           for i in range(1, self.num_workers)]}


def main(_):

    left_number = 10000
    right_number = 17321

    MyClaster = Cluster(PREFIX, NUM_WORKERS)
    cluster = tf.train.ClusterSpec(MyClaster.clust_spec)
    server = tf.train.Server(cluster,
                             job_name=MyClaster.job_name,
                             task_index=MyClaster.task_index
                             )
    # if current instance is parameter server
    if MyClaster.job_name == "ps":
        server.join()

    # for workers
    elif MyClaster.job_name == "worker":
        server.join()

    # client
    elif MyClaster.job_name == "client":
        # declare variables on parameter server
        with tf.device("job:ps/task:0/cpu:0"):
            embeddings_1 = tf.Variable(
                tf.random_uniform([right_number, embedding_size], -0.05, 0.05))
            embeddings_2 = tf.Variable(
                tf.random_uniform([left_number, embedding_size], -0.05, 0.05))
            biases = tf.Variable(tf.zeros([left_number]))

        files_names = tf.train.match_filenames_once(
                                INPUT_PATTERN, name="myFiles")
        optimizers = []

        for i in range(NUM_WORKERS-1):
            with tf.device("/job:worker/task:%d" % i):

                to_process = tf.strided_slice(files_names, [i], [999999999], strides=[NUM_WORKERS])
                filename_queue = tf.train.string_input_producer(
                    to_process, shuffle=True, num_epochs=NUM_EPOCHS
                )
                reader = tf.TextLineReader()
                _, value = reader.read(filename_queue)
                col1, col2 = tf.decode_csv(
                    value, record_defaults=[[1], [1]], field_delim="\t"
                )
                train_inputs, train_labels = tf.train.batch(
                    [col1,[col2]], batch_size=BATCH_SIZE, capacity=10*BATCH_SIZE
                )
                embed = tf.nn.embedding_lookup(embeddings_1, train_inputs)
                loss = tf.reduce_mean(
                    tf.nn.sampled_softmax_loss(
                        weights=embeddings_2,
                        biases=biases,
                        inputs=embed,
                        labels=train_labels,
                        num_sampled=7,
                        num_classes=left_number,
                        remove_accidental_hits=True))
                optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
                optimizers.append(optimizer)

        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess = tf.Session(server.target)
        sess.run(init)

        for i in range(1, NUM_WORKERS):
            with tf.device("/job:worker/task:%d" % MyClaster.task_index):
                sess.run(tf.local_variables_initializer())
                coord = tf.train.Coordinator()
                _ = tf.train.start_queue_runners(sess=sess, coord=coord)

        counter = 1
        for _ in range(MAX_STEPS):
            _ = sess.run(optimizers)
            counter += 1


if __name__ == "__main__":
    main(None)
