import pulp
from ..utils.validation import check_is_fitted


class BigDataNewsvendor:
    def __init__(self, cp, ch, lp_type):
        self.cp = 2
        self.ch = 1
        self.lp_type = 'minimize'

    def fit(self, x, y):
        # Add intercept
        x.insert(0, 'intercept', 1)

        if self.lp_type == 'minimize':
            nvAlgo = pulp.LpProblem(sense=pulp.LpMinimize)

        elif self.lp_type == 'maximize':
            nvAlgo = pulp.LpProblem(sense=pulp.LpMaximize)

        else:
            raise NameError('lp_type must be eigther minimize or maximize')

        n = x.index.values
        p = x.columns.values

        q = pulp.LpVariable.dicts('q', p)
        u = pulp.LpVariable.dicts('u', n, lowBound=0)
        o = pulp.LpVariable.dicts('o', n, lowBound=0)

        nvAlgo += (sum([self.cp * u[i] for i in n]) + sum([self.ch * o[i] for i in n])) / len(n)

        for i in n:
            nvAlgo += u[i] >= y.loc[i] - q['intercept'] - sum([q[j] * x.loc[i, j] for j in p if j != 'intercept'])
            nvAlgo += o[i] >= q['intercept'] + sum([q[j] * x.loc[i, j] for j in p if j != 'intercept']) - y[i]

        nvAlgo.solve()

        self.q_ = q

        return self

    def predict(self, X):
        check_is_fitted(self)
        X.insert(0, 'intercept', 1)
        pred = []
        for index, row in X.iterrows():
            # access data using column names
            value = 0
            for name in X.columns:
                value = value + row[name] * self.q_[name].value()
            pred += [value]
        return pred