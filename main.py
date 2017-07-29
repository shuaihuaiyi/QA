import os
import time

import tensorflow as tf

import qaData
from qaLSTMNet import QaLSTMNet


def restore():
    try:
        print("正在加载模型，大约需要一分钟...")
        saver.restore(sess, trainedModel)
    except Exception as e:
        print(e)
        print("加载模型失败，重新开始训练")
        train()


def train():
    print("重新训练，请保证计算机拥有至少8G空闲内存与2G空闲显存")
    # 准备训练数据
    print("正在准备训练数据，大约需要五分钟...")
    qTrain, aTrain, lTrain, qIdTrain = qaData.loadData(trainingFile, word2idx, unrollSteps, True)
    qDevelop, aDevelop, lDevelop, qIdDevelop = qaData.loadData(developFile, word2idx, unrollSteps, True)
    trainQuestionCounts = qIdTrain[-1]
    for i in range(len(qIdDevelop)):
        qIdDevelop[i] += trainQuestionCounts
    tqs, tta, tfa = [], [], []
    for question, trueAnswer, falseAnswer in qaData.trainingBatchIter(qTrain + qDevelop, aTrain + aDevelop,
                                                                      lTrain + lDevelop, qIdTrain + qIdDevelop,
                                                                      batchSize):
        tqs.append(question), tta.append(trueAnswer), tfa.append(falseAnswer)
    print("加载完成！")
    # 开始训练
    print("开始训练，全部训练过程大约需要12小时")
    sess.run(tf.global_variables_initializer())
    lr = learningRate  # 引入局部变量，防止shadow name
    for i in range(lrDownCount):
        optimizer = tf.train.GradientDescentOptimizer(lr)
        optimizer.apply_gradients(zip(grads, tvars))
        trainOp = optimizer.apply_gradients(zip(grads, tvars), global_step=globalStep)
        for epoch in range(epochs):
            for question, trueAnswer, falseAnswer in zip(tqs, tta, tfa):
                startTime = time.time()
                feed_dict = {
                    lstm.inputQuestions: question,
                    lstm.inputTrueAnswers: trueAnswer,
                    lstm.inputFalseAnswers: falseAnswer,
                    lstm.keep_prob: dropout
                }
                _, step, _, _, loss = \
                    sess.run([trainOp, globalStep, lstm.trueCosSim, lstm.falseCosSim, lstm.loss], feed_dict)
                timeUsed = time.time() - startTime
                print("step:", step, "loss:", loss, "time:", timeUsed)
            saver.save(sess, saveFile)
        lr *= lrDownRate


if __name__ == '__main__':
    # 定义参数
    trainingFile = "data/training.data"
    developFile = "data/develop.data"
    testingFile = "data/testing.data"
    resultFile = "predictRst.score"
    saveFile = "newModel/savedModel"
    trainedModel = "trainedModel/savedModel"
    embeddingFile = "word2vec/zhwiki_2017_03.sg_50d.word2vec"
    embeddingSize = 50  # 词向量的维度

    dropout = 1.0
    learningRate = 0.4  # 学习速度
    lrDownRate = 0.5  # 学习速度下降速度
    lrDownCount = 4  # 学习速度下降次数
    epochs = 20  # 每次学习速度指数下降之前执行的完整epoch次数
    batchSize = 20  # 每一批次处理的<b>问题</b>个数

    rnnSize = 100  # LSTM cell中隐藏层神经元的个数
    margin = 0.1  # M is constant margin

    unrollSteps = 100  # 句子中的最大词汇数目
    max_grad_norm = 5  # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小

    allow_soft_placement = True  # Allow device soft device placement
    gpuMemUsage = 0.75  # 显存最大使用率
    gpuDevice = "/gpu:0"  # GPU设备名

    # 读取测试数据
    print("正在载入测试数据，大约需要一分钟...")
    embedding, word2idx = qaData.loadEmbedding(embeddingFile)
    qTest, aTest, _, qIdTest = qaData.loadData(testingFile, word2idx, unrollSteps)
    print("测试数据加载完成")
    # 配置TensorFlow
    with tf.Graph().as_default(), tf.device(gpuDevice):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpuMemUsage)
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            # 加载LSTM网络
            print("正在加载LSTM网络，大约需要三分钟...")
            globalStep = tf.Variable(0, name="globle_step", trainable=False)
            lstm = QaLSTMNet(batchSize, unrollSteps, embedding, embeddingSize, rnnSize, margin)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars), max_grad_norm)
            saver = tf.train.Saver()
            print("加载完成！")

            # 加载模型或训练模型
            if os.path.exists(trainedModel + '.index'):
                while True:
                    choice = input("找到已经训练好的模型，是否载入（y/n）")
                    if choice.strip().lower() == 'y':
                        restore()
                        break
                    elif choice.strip().lower() == 'n':
                        train()
                        break
                    else:
                        print("无效的输入！\n")
            else:
                train()

            # 进行测试，输出结果
            print("正在进行测试，大约需要三分钟...")
            with open(resultFile, 'w') as file:
                for question, answer in qaData.testingBatchIter(qTest, aTest, batchSize):
                    feed_dict = {
                        lstm.inputTestQuestions: question,
                        lstm.inputTestAnswers: answer,
                        lstm.keep_prob: dropout
                    }
                    _, scores = sess.run([globalStep, lstm.result], feed_dict)
                    for score in scores:
                        file.write("%.9f" % score + '\n')
    print("所有步骤完成！程序结束")
