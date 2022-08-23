#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    sub = [abs(net_worths[i] - predictions[i]) for i in range(len(predictions))]
    er = list(zip(sub, net_worths, ages))
    er.sort(reverse=True)
    cleaned_data = [(k,j,i) for (i,j,k) in er[9:]]
    
    return cleaned_data

