"""
This is a data preparation library with some tools that might not have to do 
with the core library in general but with operations that might
happen more than once.
"""

def remove_random(data, percentage=.2):
    """
    Helper function to remove percentage data points from a data frame.
    The NaN values are distributed equally over the whole data set.
    """

    # shuffle data and re-index
    data = data.sample(frac=1).reset_index(drop=True)
    # determine how many data points need to be removed per column
    frac = int(percentage * len(data)) / len(data.columns)

    counter = 1
    # iterate over columns and remove frac amount of data points
    for column in data.columns:
        lo = (counter - 1) * frac
        hi = (counter * frac) - 1
        data.loc[lo:hi, column] = None
        counter += 1
    return data