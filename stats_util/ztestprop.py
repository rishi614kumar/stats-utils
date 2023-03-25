from scipy import stats

class ZtestProportions:

    def __init__(self, n1: int, y1: int, n2: int, y2: int) -> None:
        self.n1 = n1
        self.y1 = y1
        self.n2 = n2
        self.y2 = y2

    def z_test_proportions(self, alpha : float=0.05) -> tuple[float, float, float, float, bool]:
        '''
        A function to return the calculated z-score and its associated p-value
        Input: A significance level
        Output: Z-score, p-value, confidence interval bounds, sample size assumption violation flag
        '''

        n1 = self.n1
        y1 = self.y1

        n2 = self.n2
        y2 = self.y2
        
        p1 = y1 / n1

        p2 = y2 / n2

        p = (y1 + y2) / (n1 + n2)

        self.z_n = (p1 - p2) / ((p * (1 - p) ** (1/2)) * (((1 / n1) + (1 / n2)) ** (1/2)))

        # One-tailed p-value
        self.p_value = (stats.norm.sf(abs(z_n)))
        
        # Find the critical value for the specified significance level
        self.critical_value = stats.norm.pff(1-alpha)

        # Calculate the standard error under the alternative hypnothesis
        self.SEa = (p1 * (1 - p1) / n1 + p2 (1 - p2) / n2) ** (1/2)

        # Create a boolean to represent whether or not the sample size assumption was violated
        LEAST_SAMPLES = 5
        self.ss_assump = ((p1 * n1 > LEAST_SAMPLES) and ((1 - p1) * n1 > LEAST_SAMPLES)) and ((p2 * n2 > LEAST_SAMPLES) and ((1 - p2) * n2 > LEAST_SAMPLES))

        # Calculate the confidence interval for the difference in proportions
        self.ci_minus = (p1 - p2) - (self.SEa * self.critical_value)
        self.ci_plus = (p1 - p2) + (self.SEa * self.critical_value)

        return self.z_n, self.p_value, self.ci_minus, self.ci_plus, self.ss_assump




