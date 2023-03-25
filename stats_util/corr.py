from scipy import stats
import pandas as pd
from math import log, exp, sqrt

class Correlation:

    def __init__(self, x: pd.Series, y: pd.Series, tails: str, pearson: str = True) -> None:
        valid_tails = ['left', 'right', 'both']
        if tails.lower() not in valid_tails:
            raise ValueError('Bad input for \'tails\'')
        
        assert len(x) == len(y)

        if tails == 'both':
            self.alt = 'two-sided'
            self.tails = 2
        elif tails == 'left':
            self.alt = 'less'
            self.tails = 1
        elif tails == 'right':
            self.alt == 'greater'
            self.tails = 1
        
        self.x = x.values.reshape(-1, 1)
        self.y = y.values.reshape(-1, 1)
        self.pearson = pearson

    def correlation(self) -> tuple[float, float]:
        if self.pearson:
            u1 = self.x.mean()
            u2 = self.y.mean()
            sig1 = self.x.std()
            sig2 = self.y.std()

            
            corr = ((self.x * self.y).mean() - u1 * u2) / (sig1 * sig2)
            # n - 2 degrees of freedom
            p_value = stats.t.sf(abs(self.t_statistic_corr()), df = len(self.x) - 2) * self.tails
        else:
            corr, p_value = stats.spearmanr(self.x, self.y, alternative=self.alt)
        
        return corr, p_value

    def t_statistic_corr(self) -> float:
        r = self.corr
        n = len(self.x)

        # n - 2 degrees of freedom
        t_stat = (r * ((n - 2) ** (1/2))) / ((1 - r ** 2) ** (1/2))

        return t_stat
    
    def get_confidence_interval(self, alpha : float) -> None:
        r = self.corr
        n = len(self.x)
        r_fish = log((1 + r) / (1 - r)) / 2

        SE = 1 / sqrt(n - 3)

        # critical value for one tailed test (divide alpha by 2 for two tailed)
        critical_value = stats.norm.ppf(1 - alpha / self.tails)

        lower_bound = r_fish - critical_value * SE
        upper_bound = r_fish + critical_value * SE

        self.ci_lower = (exp(2 * lower_bound) - 1) / (exp(2 * lower_bound) + 1)
        self.ci_upper = (exp(2 * upper_bound) - 1) / (exp(2 * upper_bound) + 1)

    def t_test(self, alpha: float=0.05, verbose=True) -> bool:
        sig = False
        if self.pearson:
            if verbose: print('Pearson')
            keyword = 'linear '
        else:
            if verbose: print('Spearman Rank')
            keyword = ''
        self.corr, self.p_value = self.correlation()

        significance_message = 'The correlation (%s) is %ssignificant with p=%s and alpha %s'
        evidence_message = '%s evidence to conclude that there is a %s' + str(keyword) + 'relationship between x and y because the correlation coefficient is %s zero with significance'

        if self.tails == 1:
            if self.alt == 'greater':
                if verbose: print('Right Tailed')
                if self.p_value < alpha and self.corr > 0:
                    significance = ''
                    if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                    sufficiency = 'Sufficient'
                    posneg = 'positive '
                    distance = 'greater than'
                    if verbose: print(evidence_message % (sufficiency, posneg, distance))
                    sig = True
                else:
                    significance = 'NOT '
                    if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                    sufficiency = 'Insufficient'
                    posneg = 'positive '
                    distance = 'NOT greater than'
                    if verbose: print(evidence_message % (sufficiency, posneg, distance))
                    sig = False
            else:
                if verbose: print('Left Tailed')
                if self.p_value < alpha and self.corr < 0:
                    significance = ''
                    if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                    sufficiency = 'Sufficient'
                    posneg = 'negative '
                    distance = 'less than'
                    if verbose: print(evidence_message % (sufficiency, posneg, distance))
                    sig = True
                else:
                    significance = 'NOT '
                    if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                    sufficiency = 'Insufficient'
                    posneg = 'negative '
                    distance = 'NOT less than'
                    if verbose: print(evidence_message % (sufficiency, posneg, distance))
                    sig = False
        else:
            if verbose: print('Two-tailed Test')
            if self.p_value < alpha:
                significance = ''
                if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                sufficiency = 'Sufficient'
                posneg = ''
                distance = 'different from'
                if verbose: print(evidence_message % (sufficiency, posneg, distance))
                sig = True
            else:
                significance = 'NOT '
                if verbose: print(significance_message % (self.corr, significance, self.p_value, alpha))
                sufficiency = 'Insufficient'
                posneg = ''
                distance = 'NOT different from'
                if verbose: print(evidence_message % (sufficiency, posneg, distance))
                sig = False
        if verbose: print('Correlation:', self.corr)

        if len(self.x) > 1 and self.pearson:
            self.get_confidence_interval(alpha=alpha)
            if verbose: print(str(int((1 - alpha) * 100)) + '% Confidence Interval: [' + str(self.ci_lower) + ', ' + str(self.ci_upper) + ']')
        if verbose: print('Samples:', len(self.x))
        return sig, self.corr, self.p_value


