#coding=utf-8
import tensorflow as tf

from config import Config
from data_processor import DataSet
from model import CharCNN

flags = tf.app.flags

# input args
flags.DEFINE_string("train_data_path", None, "Train data path")
flags.DEFINE_string("validate_data_path", None, "Validate data path")
FLAGS = flags.FLAGS

def main(unused_argv):
    train_data_path = FLAGS.train_data_path
    val_data_path = FLAGS.validate_data_path

    # load train data
    train_data = DataSet(train_data_path)
    dev_data = DataSet(val_data_path)
    train_data.dataset_process()
    dev_data.dataset_process()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,        log_device_placement=False)
        sess = tf.Session(config = session_conf)
        with sess.as_default():
            cnn = CharCNN(
                l0 = Config.l0,
                num_classes = Config.nums_classes,
                conv_layers = Config.model.conv_layers,
                fc_layers = Config.model.fc_layers,
                l2_reg_lambda = 0
                )
            global_step = tf.Variable(0, name = 'global_step', trainable = False)
            optimizer = tf.train.AdamOptimizer(Config.model.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: Config.model.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        print "初始化完毕，开始训练"
        for i in range(Config.training.epoches):
            batch_train = train_data.next_batch()
            # 训练模型
            train_step(batch_train[0], batch_train[1])
            current_step = tf.train.global_step(sess, global_step)
            # train_step.run(feed_dict={x: batch_train[0], y_actual: batch_train[1], keep_prob: 0.5})
            # 对结果进行记录
            if current_step % Config.training.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(dev_data.doc_image, dev_data.label_image, writer=dev_summary_writer)
                print("")
            if current_step % Config.training.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

if __name__ == "__main__":
    tf.app.run()