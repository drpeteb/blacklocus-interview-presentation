import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt

def simulateData(meanWeeklySales, weeklyPattern, numWeeks, exclusionRate):
    sales = np.zeros(7*numWeeks)
    weeklySales = np.zeros(numWeeks)
    thisWeeklySales = meanWeeklySales
    for ww in range(numWeeks):
        thisWeeklySales *= stats.norm.rvs(loc=1,scale=0.1)
        weeklySales[ww] = thisWeeklySales
        for dd in range(7):
            sales[7*ww+dd] = \
                           stats.poisson.rvs(thisWeeklySales*weeklyPattern[dd])
            if stats.bernoulli.rvs(exclusionRate):
                sales[7*ww+dd] = np.nan
    return sales, weeklySales

def modelLhood(sales, weeklyPattern, weeklySales):
    predSales = np.outer(weeklyPattern, weeklySales)
    logLhood = np.nansum( -predSales + sales*np.log(predSales) - sp.special.gammaln(1+sales) )
    return logLhood

def trainModel(sales):
    numWeeks = sales.shape[0]/7
    sales = np.reshape(sales,(7,numWeeks),order='F')
    mask = np.logical_not(np.isnan(sales))
    weeklyPattern = np.ones(7)/7
    weeklySales = 7*np.nanmean(sales)*np.ones(numWeeks)
    old_logLhood = modelLhood(sales, weeklyPattern, weeklySales)
    change = np.inf
    while change > 0.1:
        weeklyPattern = np.nansum(sales,axis=1)/np.sum(mask*weeklySales[np.newaxis,:],axis=1)
        weeklyPattern /= np.sum(weeklyPattern)
        weeklySales = np.nansum(sales,axis=0)/np.sum(mask*weeklyPattern[:,np.newaxis],axis=0)
        logLhood = modelLhood(sales, weeklyPattern, weeklySales)
        change = logLhood-old_logLhood
        old_logLhood = logLhood
        print(logLhood)
    return weeklyPattern, weeklySales
    

plt.close('all')
np.random.seed(2)

# Pattern
weeklyPattern = np.array([0.12,0.08,0.08,0.12,0.15,0.35,0.10])

# Fast seller
fastSellerSales, fastSellerWeekly = simulateData(350, weeklyPattern, 10, 0.05)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fastSellerSales, 'k')
ax.set_xlabel('day')
ax.set_ylabel('sales')
fig.savefig('fast_seller.pdf', bbox_inches='tight')

# Medium seller
mediumSellerSales, mediumSellerWeekly = simulateData(35, weeklyPattern, 10, 0.05)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(mediumSellerSales, 'k')
ax.set_xlabel('day')
ax.set_ylabel('sales')
fig.savefig('medium_seller.pdf', bbox_inches='tight')

# Slow seller
slowSellerSales, slowSellerWeekly = simulateData(3.5, weeklyPattern, 10, 0.05)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(slowSellerSales, 'k')
ax.set_xlabel('day')
ax.set_ylabel('sales')
fig.savefig('slow_seller.pdf', bbox_inches='tight')

# Fast again with different size figure
fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(1,1,1)
ax.plot(fastSellerSales, 'k')
ax.set_xlabel('day')
ax.set_ylabel('sales')
fig.savefig('fast_seller_small.pdf', bbox_inches='tight')

# Factorisation
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(weeklyPattern, 'r')
ax.set_ylim([0,0.4])
ax.set_xlabel('day of the week')
ax.set_ylabel('proportion')
ax = fig.add_subplot(1,2,2)
ax.plot(fastSellerWeekly, 'g')
ax.set_ylim([0,500])
ax.set_xlabel('week')
ax.set_ylabel('sales')
plt.tight_layout()
fig.savefig('fast_factorisation.pdf', bbox_inches='tight')

# Reconstruction
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fastSellerSales, 'k')
ax.plot( np.outer(weeklyPattern,fastSellerWeekly).T.flatten() )
fig.savefig('fast_reconstruction.pdf', bbox_inches='tight')

# Training - fast
trainedFastWeeklyPattern, trainedFastWeeklySales = trainModel(fastSellerSales)
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(weeklyPattern, 'r')
ax.plot(trainedFastWeeklyPattern, 'r:')
ax.set_ylim([0,0.4])
ax.set_xlabel('day of the week')
ax.set_ylabel('proportion')
ax = fig.add_subplot(1,2,2)
ax.plot(fastSellerWeekly, 'g')
ax.plot(trainedFastWeeklySales, 'g:')
ax.set_ylim([0,500])
ax.set_xlabel('week')
ax.set_ylabel('sales')
plt.tight_layout()
fig.savefig('fast_learning.pdf', bbox_inches='tight')

# Training - medium
trainedMediumWeeklyPattern, trainedMediumWeeklySales = trainModel(mediumSellerSales)
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(weeklyPattern, 'r')
ax.plot(trainedMediumWeeklyPattern, 'r:')
ax.set_ylim([0,0.4])
ax.set_xlabel('day of the week')
ax.set_ylabel('proportion')
ax = fig.add_subplot(1,2,2)
ax.plot(mediumSellerWeekly, 'g')
ax.plot(trainedMediumWeeklySales, 'g:')
ax.set_ylim([0,50])
ax.set_xlabel('week')
ax.set_ylabel('sales')
plt.tight_layout()
fig.savefig('medium_learning.pdf', bbox_inches='tight')

# Training - slow
trainedSlowWeeklyPattern, trainedSlowWeeklySales = trainModel(slowSellerSales)
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1,2,1)
ax.plot(weeklyPattern, 'r')
ax.plot(trainedSlowWeeklyPattern, 'r:')
ax.set_ylim([0,0.4])
ax.set_xlabel('day of the week')
ax.set_ylabel('proportion')
ax = fig.add_subplot(1,2,2)
ax.plot(slowSellerWeekly, 'g')
ax.plot(trainedSlowWeeklySales, 'g:')
ax.set_ylim([0,5])
ax.set_xlabel('week')
ax.set_ylabel('sales')
plt.tight_layout()
fig.savefig('slow_learning.pdf', bbox_inches='tight')