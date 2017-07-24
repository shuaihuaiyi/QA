import readData
import evaluation
import tensorflow as tf

if __name__ == '__main__':
    # 定义参数
    trainingFile = "data/training.data"
    testFile = "data/develop.data"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50

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
