import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
import cx_Oracle as oracle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 忽略警告
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'




label_map = {'起草': 1, '采购经理': 2, '采购总监': 3, '财务总监': 4, '采购VP': 5, 'CEO审批': 6,
             'CEO手工单': 7, '特控部备案': 8,'采购经理(音膜)' :9,'采购经理（音膜）' :9,'财务审核（零星）' :10,
            '采购总监(音膜)':11,'采购经理（标准）':12,'价格组经理/采购经理审批）':13,
              '价格总监(音膜)':14, '财务审核':15 , '价格总监':16 , '价格组经理/采购经理审批':17}

label_map1 = {'null': 1, '月结30天': 2, '月结45天': 3, '月结60天': 4, '月结90天': 5, '月结180天': 6,
             '入帐后30天': 7, '入帐后45天': 8, '入帐后60天':9, '入帐后75天': 10, '入帐后90天': 11, '入账后120天': 12,
             '货、票到7天付款': 13, '货、票到15天付款': 14, '货、票到30天付款': 15, '货、票到45天付款': 16,
              '(停用)月结30天': 8, '货到付款': 8, '90天远期L/C': 8, '预付': 8, 'Net 30 Days To HK YEC': 8}

label_map2 ={'国内': 1, '国外': 2}

print(label_map1.get('入账后120天'))


def oracle_data():
    conn = oracle.connect('csmart/csmart@192.168.1.69:1521/smartformsdb')
    cursor = conn.cursor()
    sql = "select  f.text1 , f.document_id  ,f.zje,f.currentsapfactory   from FORM_PU01 f  where  " \
          "f.document_id  in (select document_id from PU01_EXPORT_TABLE)" \
          " and f.currentsapfactory is not null and f.text1 is not null and f.zje is not null" \
          "   "

    cursor.execute(sql)

    cursorV = conn.cursor()

    x_data = np.zeros(3)
    y_row = np.zeros(1)   #y目前只存放环节名称
    y_data = np.zeros(1)  #zeros(1)创建长度1的全0数组(这个是一维的只有一行)
    x_row = np.zeros(3)   #x目前只存放金额

    # 堆叠数组：stack()，hstack()，vstack()
    for result in cursor:  # 循环从游标获取每一行并输出该行。


        #此处可优化，合并同一sql，后期可优化
        sql = "select  task_name,record_id  from  ( " \
              "select aa.* ,  RANK() OVER(PARTITION BY aa.document_id  ORDER BY  aa.create_time ) a  from (" \
              "select tt.*, RANK() OVER(PARTITION BY tt.document_id,  tt.SRC_NODE_ID" \
              " ORDER BY  tt.create_time ) sort2 from (" \
              "select task_name, SRC_NODE_ID, document_id ,record_id, create_time ," \
              "RANK() OVER(PARTITION BY document_id ORDER BY  create_time asc) sort " \
              "from PU01_EXPORT_TABLE where EXCHANGE_TYPE='submit' and document_id='"+result[1]+"'" \
              "order by document_id , create_time  asc) tt  order by document_id ,create_time asc ) aa " \
              " where sort2 = 1) where a=2"

        cursorV.execute(sql)

        #for resultV in cursorV:
        #print(cursorV.fetchone()[0])
        #目标只有一条，只取第一条
        all=cursorV.fetchall()
        if len(all)>0:
            #当经过第二个环节得文档才计入集合中
            x_row[0] = label_map1.get(result[0])
            x_row[1] = result[2]
            x_row[2] = label_map2.get(result[3])
            x_data = np.row_stack((x_data, x_row))  # 两个数组相加：加行


            print(result[0])
            print(label_map1.get(result[0]))

            print(result[3])
            print(label_map2.get(result[3]))

            print(all[0][0])
            print(label_map.get(all[0][0]))
            y_row[0] = label_map.get(all[0][0])
            #y_row = hash(a[1])
            y_data = np.append(y_data, y_row)  # 一个数组扩展：加列

    x_data = x_data[1:len(x_data)]
    y_data = y_data[1:len(y_data)]
    # 关闭游标、oracle连接
    cursor.close()
    cursorV.close()
    conn.close()

    # 数据整理
   # y_data = np.delete(y_data, [len(y_data)-1])
    print(x_data)
    print(y_data)
    return x_data, y_data


# 定义随机batch：在训练数据中，随机取batch_size 个样本
def random_batch(x_train, y_train, batch_size):
    # 随机生成数组下标，以便取数
    rnd_indices = np.random.randint(0, len(x_train), batch_size)
    x_train = x_train[rnd_indices]
    y_train = y_train[rnd_indices]
    return x_train, y_train


# 建立全连接神经网络层：三个隐层，激活函数为ELU、L2正则化
def fc_layers(input_tensor, regularizer):
    hindenn1 = 42
    hindenn2 = 28
    with tf.name_scope("full-connect-layer"):
        fc1 = tf.layers.dense(input_tensor, hindenn1, activation=tf.nn.elu,
                              kernel_regularizer=regularizer, name="fc1")
        # 使用Softmax 概率相加为1。Sigmoid 的概率和不为0，没有可加性，影响优化速度。
        fc2 = tf.layers.dense(fc1, hindenn2, activation=tf.nn.softmax,
                              kernel_regularizer=regularizer, name="fc2")

    return fc2


def train(data, label, learning_rate, lambd, n_epoch, batch_size):
    # 划分训练集、测试集
    test_radio = 0.9
    test_size = int(len(data) * test_radio)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
    train_label = label[:-test_size]
    test_label = label[-test_size:]

    # 搭建模型结构：定义输入数据、输出类别
    n_input = train_data.shape[1]  # 输入数量
    n_output = max(train_label)+1 #len(set(train_label))  # 输出类别（去重）：14 个标签
    print("标签: ")
    print(set(train_label))
    print("n_output", n_output)
    # 建立输入层，使用预定义的全连接神经网络。
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=(None, n_input), name="x")  # 占位符,根据输入的数据类型
        y = tf.placeholder(tf.int32, shape=None, name="y")  # 占位符,根据输出的数据类型

    # regularizer = None  # 不加正则化：在noise变大时，效果就变差了，泛化能力较差。
    regularizer = tf.contrib.layers.l2_regularizer(lambd)  # 加入正则化，noise变大时，泛化效果有提升。
    # 报错：ValueError: An initializer for variable fc1/kernel of <dtype: 'string'> is required
    fc2 = fc_layers(x, regularizer)  # 可能要将文本转换为向量？？？
    # 建立输出层，也使用全连接，将输入层结果与输出类别连接起来。
    with tf.name_scope("output"):
        logits = tf.layers.dense(fc2, n_output, kernel_regularizer=regularizer, name="output")

    # 使用稀疏矩阵交叉熵计算损失，建立损失节点。
    # summary是对网络中Tensor取值进行监测的一种Operation。这些操作在图中是“外围”操作，不影响数据流本身。
    # SummaryWriter文件中存储的是序列化的结果，需要借助TensorBoard才能查看。
    with tf.name_scope("loss"):
        # 使用softmax输出层时，选择交叉熵
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)  # 归一化指数函数
        loss = tf.reduce_mean(xentropy, name="loss")
        loss_summary = tf.summary.scalar("loss", loss)  # 添加标量统计结果：将每一步的loss写入。

    # 建立训练节点。
    global_step = tf.Variable(0, trainable=False)  # Variable集合：模型参数
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)  # 使用Adam 优化算法
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    # 建立评价节点。
    with tf.name_scope("eval"):
        predictions = tf.argmax(logits, 1)  # 预测输出结果
        correct = tf.nn.in_top_k(logits, y, 1)  # 预测结果正确个数
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 计算准确率
        acc_summary = tf.summary.scalar("acc", accuracy)  # 将准确率实时写入监控

    summary_op = tf.summary.merge([loss_summary, acc_summary])  # 将需要写入文件的值merge
    # 保存模型结果、日志、中断的断点epoch。
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = './logs/' + now          # 日志文件路径
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())  # 将Graph对象写入磁盘log文件
    saver = tf.train.Saver()  # 生成模型的saver

    # 训练和验证模型
    n_batches = int(np.ceil(len(data) / batch_size))
    with tf.Session() as sess:
        # 初始化全局变量：返回一个全局变量list
        init = tf.global_variables_initializer()
        # 初始化开始epoch：根据历史检查点文件里是否存在中断的epoch数。
        start_epoch = 0
        sess.run(init)  # session run之前，图中的全部Variable必须被初始化

        for epoch in range(start_epoch, n_epoch):
            for batch_index in range(n_batches):
                x_batch, y_batch = random_batch(train_data, train_label, batch_size)  # 取一个批次的训练数据
                sess.run(train_op, feed_dict={x: x_batch, y: y_batch})  # 训练过程
            loss_val, summary_str, test_pred, test_acc = sess.run(
                                            [loss, summary_op, predictions, accuracy],
                                            feed_dict={x: train_data, y: train_label})

            file_writer.add_summary(summary_str, epoch)  # 将每一个epoch的写入日志文件
            # 每 50 epoch 打印测试集上的损失和正确率
            if (epoch+1) % 10 == 0:
                print("Epoch:", epoch+1, "\tLoss:", loss_val, "\tAcc:", test_acc)

        pred_label = predictions.eval(feed_dict={x: test_data}) # 对测试集进行预测：得到正确的标签。
        jayson_label = predictions.eval(feed_dict={x: [[11,18000,1], [11,280000,1], [9,2400,1]]})
        # print(test_data)
        print(jayson_label)
        # print(pred_label)
        # print('precision_score：', precision_score(test_label, pred_label, average='weighted', labels=[2,3,4,6]))  # 测试集预测的准确率
        # print('recall_score：', recall_score(test_label, pred_label, average='weighted'))  # 测试集预测的召回率
        sess.close()


if __name__ == '__main__':
    X_data, y_data = oracle_data()
    # 使用pandas库的图表来更好的展示训练集中的几组数据
    #column_names = ['金额']
    #df = pd.DataFrame(X_data, columns=column_names)
    #print(df.head())



    learning_rate = 0.01  # 学习率
    lambd = 0.01  # 正则化率
    n_epochs = 100
    batch_size = 500
    train(X_data, y_data, learning_rate, lambd, n_epochs, batch_size)

# 命令行启动tensorboard，可视化以保存的图。
# tensorboard --logdir=D:\pycharm\codeTest\mlTest\logs