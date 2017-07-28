import readData
import taevaluation
import tensorflow as tf

if __name__ == '__main__':
    # 定义参数
    trainingFile = "data/training.data"
    validFile = "data/develop.data"
    testFile = "data/testing.data"
    saveFile = "savedModel"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50 #词向量的维度

    dropout = 1.0
    learningRate = 0.4
    batchSize = 20  # 每一批次处理的问题个数
    epochs = 20
    tf.flags.DEFINE_integer("rnn_size", 100, "rnn size")
    tf.flags.DEFINE_integer("num_rnn_layers", 1, "embedding size")
    tf.flags.DEFINE_integer("num_unroll_steps", 100, "句子中的最大词汇数目")
    tf.flags.DEFINE_integer("max_grad_norm", 5, "max grad norm")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_float("gpu_options", 0.75, "use memory rate")

    gpuMemUsage = 0.8
    gpuDevice = "/gpu:0"

    # 读取数据
    trainingList = readData.readFile(trainingFile)
    testList = readData.readFile(testFile)
    embeddingDict = readData.readEmbeddingFile(embeddingFile, embeddingSize)

    # 预处理
    trainingVec = readData.textToVec(trainingList, embeddingDict)
    testVec = readData.textToVec(testList, embeddingDict)
    del embeddingDict  # 减少内存占用

    # 定义模型 todo

    # 开始训练
    with tf.Graph().as_default():
        with tf.device(gpuDevice):
            gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=gpuMemUsage)
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=gpuOptions)
            with tf.Session(config=session_conf).as_default() as sess:
                pass  # todo

    # 评估 todo
    pass
