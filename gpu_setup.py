import tensorflow as tf

def setup_gpu():
    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found, using CPU instead.")
        return tf.keras.mixed_precision.experimental.Policy('float32')

    # Set up the GPU environment
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Enable mixed-precision training
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

    return policy
