import math
import matplotlib.pyplot as plt

## TODO:
# Assert function

class Gaussian():
    """
    Gaussian distribution class for claculating and visualizing a
    Gaussian distribution

    Attibutes:
    """
    def __init__(self, mu = 0, sigma = 1):
        self.mean = mu
        self.stdev = sigma
        self.data = []

    def calculate_mean(self) -> float:
        """
        MEthod to calculate the mean of the data set

        Args:
            None
        Returns:
            float: mean of the data set
        """
        avg = 1.0 * sum(self.data) / len(self.data)
        return avg

    def calculate_var(self):
        mean = self.calculate_mean()
        var = sum((self.data - mean)**2) / len(self.data)
        return var

    def calculate_std(self):
        std = math.sqrt(calculate_var())
        return std

    def read_data_file(self, file_name, sample = True):
        """
        Method to read in data from a txt file. The txt file should have one
        number (floa) per line. the number asre stored in the data attribute.
        After reading in the file, the mean and standard deviation
        are calculated.

        Args:
            filename (string): path to the file to read
            sample (boolean): No idie
        Returns:
            None
        """

        with open(file_name, 'r') as file:
            data_list = []
            line = file.readline()
            while line:
                data_list.append(line)
                line = file.readline()
        file.close()

        # update the class attributes
        self.data = data_list
        self.mean = calculate_mean()
        self.stad = calculate_std()


    def plot_histogram(self):
        plt.histogram(self.data)

    def pdf(self, x);
        """
        Probability density function claculator for the gaussian distribution

        Args:
            x (float): point for calculating the probability density function
        Returns:
            float: probability density function output
        """
        mu = self.mean()
        sigma = self.std()

        p = (1 / (2*math.PI*sigma)) * (- math.E * ((x - mu)**2)/ 2 * sigma**2)
        return p

    
