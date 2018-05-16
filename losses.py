def sum_of_losses(y_true, y_pred):
    start_loss = losses.categorical_crossentropy(y_true[0], y_pred[0])
    end_loss = losses.categorical_crossentropy(y_true[1], y_pred[1])
    return start_loss + end_loss
