# Naive-Bayes

Run this program by providing the following arguments:   
1.--no-cv: Without CV.    
2.k: the number of intervals after discretizing continuous values.    
3.m: for the m-estimate. If this value is negative, use Laplace smoothing. Note that m=0 is maximum likelihood estimation.   
   
For example:   
python nbayes.py 440data\volcanoes --no-cv 10 0.01   
python nbayes.py 440data\volcanoes 5 0.01   
