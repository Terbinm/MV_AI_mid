import tensorflow as tf

# 列出可用的GPU設備
print("GPU可用設備: ", tf.config.list_physical_devices('GPU'))

# 列出TensorFlow是否可以看到GPU
print("TensorFlow是否可以看到GPU: ", tf.test.is_gpu_available())

# TensorFlow 2.x使用的GPU檢測方法
print("使用TensorFlow運算的設備: ", tf.debugging.set_log_device_placement(True))

# 簡單測試GPU運算
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
    print(c)