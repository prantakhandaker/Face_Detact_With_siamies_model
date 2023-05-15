import tensorflow as tf


def contrastive_loss(margin=1):
    """
    Computes the contrastive loss between the embeddings of the anchor and positive/negative images.
    Args:
        margin: Margin value to use in the loss function (default=1).
    Returns:
        contrastive_loss_fn: Contrastive loss function.
    """

    def contrastive_loss_fn(y_true, y_pred):
        """
        Contrastive loss function.
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        Returns:
            contrastive_loss: Contrastive loss value.
        """
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        loss = tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
        return loss

    return contrastive_loss_fn
