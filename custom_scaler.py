import tensorflow as tf

class TensorFlowScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = tf.constant(feature_range, dtype=tf.float32)
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = tf.cast(X, tf.float32)
        min_ = tf.reduce_min(X, axis=0)
        max_ = tf.reduce_max(X, axis=0)
        self.min_ = min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (max_ - min_)
        return self

    def transform(self, X):
        X = tf.cast(X, tf.float32)
        X_scaled = self.scale_ * (X - self.min_) + self.feature_range[0]
        return X_scaled

    def inverse_transform(self, X):
        X = tf.cast(X, tf.float32)
        X_unscaled = (X - self.feature_range[0]) / self.scale_ + self.min_
        return X_unscaled