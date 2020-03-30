

def cost(cp, ch, Y_pred, Y_true):
    if Y_pred > Y_true:
        cost = (Y_pred - Y_true) * ch
    else:
        cost = (Y_true - Y_pred) * cp
    return cost


def costs(cp, ch, Y_pred, Y_true):
    costs = []
    for i in Y_pred:
        costs[i] = cost(cp, ch, Y_pred[i], Y_true[i])
    return costs


def avg_costs(cp, ch, Y_pred, Y_true):
    avg_costs = 0
    for i in Y_pred:
        avg_costs += cost(cp, ch, Y_pred[i], Y_true[i])
    return avg_costs