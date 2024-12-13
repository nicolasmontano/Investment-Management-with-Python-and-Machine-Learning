# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 21:45:34 2020

@author: nomic
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize


'''
------------------------------------DATA BASES------------------------------------------------------
'''

'''
-------------------------------------------------------
-------------------b------------------------------
-------------------------------------------------------
'''


def annual_ret(r,periods_per_year):
    n_periods=r.shape[0]
    return ((1+r).prod())**(periods_per_year/n_periods)-1

def annual_vol(r,periods_per_year):
    ddof=0
    return r.std(ddof=ddof)*(periods_per_year**0.5) 

def sharpe_ratio(r,rf,periods_per_year):
    '''
    Sharpe ratio
    rf (annual)
    '''
    n_periods=len(r)
    rf_period=(1+rf)**(1/n_periods)-1
    
    excess_return =r-rf_period
    ann_excess=annual_ret(excess_return,periods_per_year)
    ann_vol=annual_vol(excess_return,periods_per_year)
    return ann_excess/ann_vol
        

def drawdown(returns: pd.Series):    
    '''
    Takes a series and return a df with the wealth index, peaks and teh Drawdown
    '''
    wealth_index=1000*(1+returns).cumprod()
    previous_peak=wealth_index.cummax()
    drawdowns=(wealth_index-previous_peak)/previous_peak
    return pd.DataFrame({
        'Wealth':wealth_index,
        'Peaks':previous_peak,
        'Drawdown':drawdowns
        })

def semideviation(df):
    '''
    Receives a df and returns 
    '''
    mask=df<0
    return df[mask].std(ddof=0)

def skewness(r):
    '''
    Receives a df and returns 
    '''
    demeaned_r=r-r.mean()
    #population standard deviation and not sample
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**3).mean()
    
    return exp/sigma_r**3

def kurtosis(r):
    '''
    Receives a df and returns 
    '''
    demeaned_r=r-r.mean()
    #population standard deviation and not sample
    sigma_r=r.std(ddof=0)
    exp=(demeaned_r**4).mean()
    
    return exp/sigma_r**4

#def is_normal(r,level=0.01):
#    '''
#    Receives a df and returns 
#    '''
#    statistic,p_value=scipy.stats.jarque_berra(r)
#   
#    return p_value>level

def var_historic(r,level=5):
    '''
    Receives a df and returns VaR
    '''
    if isinstance(r,pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    if isinstance(r,pd.Series):
        return -np.percentile(r,level)
    else:
        raise TypeError('Expected a df or series')

def var_gaussian(r,level=5,modified=False):
    '''
    Receives a df and returns VaR
    '''
    z=norm.ppf(level/100)
    if modified:
        s=skewness(r)
        k=kurtosis(r)
        z=(
            z+(z**2-1)*s/6+(z**3-3*z)*(k-3)/24-(2*z**3-5*z)*(s**2)/36
           )
    
    var=-(r.mean()+z*r.std(ddof=0))
    return var

'''
-------------------------------------------------------
-------------------b------------------------------
-------------------------------------------------------
'''


def portfolio_return(weights,returns):
    '''
    weights and returns -> portfolio return
    '''
    return weights.T @ returns

def portfolio_vol(weights,covmat):
    '''
    weights(n,1) and covmat(n,n) -> portfolio vol
    '''
    return (weights.T @ covmat @ weights)**0.5


def minimize_vol(target_return,er,cov):

    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    return_is_target={
        'type':'eq',
        'fun':lambda weights:target_return-portfolio_return(weights,er)
        }
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights)-1
        }

    results=minimize(portfolio_vol,
                     init_guess,
                     args=(cov,),
                     method='SLSQP',
                     options={'disp':False},
                     constraints=(return_is_target,weights_sum_to_1),
                     bounds=bounds
                     )
    return results.x

def optimal_weights(n_points,er,cov): 
    target_rs=np.linspace(er.min(),er.max(),n_points)
    weights=[minimize_vol(target_return,er,cov) for target_return in target_rs]
    return weights


def msr(riskfree_rate,er,cov):
    
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds=((0.0,1.0),)*n
    weights_sum_to_1={
        'type':'eq',
        'fun':lambda weights:np.sum(weights)-1
        }
    
    def neg_sharpe_ratio(weights,riskfree_rate,er,cov):
        r=portfolio_return(weights,er)
        vol=portfolio_vol(weights,cov)
        return -(r-riskfree_rate)/vol
        
    results=minimize(neg_sharpe_ratio,
                     init_guess,
                     args=(riskfree_rate,er,cov,),
                     method='SLSQP',
                     options={'disp':False},
                     constraints=(weights_sum_to_1),
                     bounds=bounds
                     )
    return results.x


def gmv(cov):
    n=cov.shape[0]
    return msr(0,np.repeat(1,n),cov)


def plot_ef(n_points,er,cov,riskfree_rate=0,show_gmw=False,show_ew=False,show_cml=False):
    style='.-'
    #show_ew=False
    #show_cml=False
    #show_gmw=False 
    
    '''
    plots ef    
    '''
    weights=optimal_weights(n_points,er,cov)
    rets=[portfolio_return(w,er) for w in weights]
    vols=[portfolio_vol(w,cov) for w in weights]
    
    ef=pd.DataFrame({'Returns':rets,'Volatility':vols})
    ax=ef.plot.line(x='Volatility',y='Returns',style=style,title='Efficient frontier')
    
    if show_ew:
        n=er.shape[0]
        w_ew=np.repeat(1/n,n)
        r_ew=portfolio_return(w_ew,er)
        vol_ew=portfolio_vol(w_ew,cov)
        #display EW
        ax.plot([vol_ew],[r_ew],color='goldenrod',marker='o',markersize=12,linewidth=1,label='EW')
        ax.legend()
        
    if show_cml:
        ax.set_xlim(left=0)
        w_msr=msr(riskfree_rate,er,cov)
        r_msr=portfolio_return(w_msr,er)
        vols_msr=portfolio_vol(w_msr,cov)
        
        #add CML
        cml_x=[0,vols_msr]
        cml_y=[riskfree_rate,r_msr]        
        ax.plot(cml_x,cml_y,color='green',marker='o',linestyle='dashed',markersize=8,linewidth=1,label='CML')
        
    if show_gmw:
        w_gmv=gmv(cov)
        r_gmv=portfolio_return(w_gmv,er)
        vol_gmv=portfolio_vol(w_gmv,cov)
        #display EW
        ax.plot([vol_gmv],[r_gmv],color='midnightblue',marker='o',markersize=8,linewidth=1,label='GMW')
        
    ax.legend(loc='lower right')
    ax.set_ylabel('Return')
    
    return ax


'''
-------------------------------------------------------
------------------b------------------------------
-------------------------------------------------------
'''
def summary_stats(r,rf,periods_per_year=12):    
    '''
    Summary of basic statistics    
    rf=annual risk free rate
    '''
    #days_year=252 
    #months_year=12 
    ddof=0 # population degrees od freedom
    
    if not (isinstance(r, pd.DataFrame) or isinstance(r, pd.Series) ): 
        raise Exception('expected a pd.Series or pd.DataFrame')
        
    if isinstance(r, pd.Series):
        r=pd.DataFrame(r,columns=["r"])
        
    n_periods=r.shape[0]
    
    a_ret=annual_ret(r,periods_per_year)
    a_vol=annual_vol(r,periods_per_year)
    s_ratio=sharpe_ratio(r,rf,periods_per_year)
    Max_drawdown=r.aggregate(lambda x:drawdown(x).loc[:,'Drawdown'].min())
    skw=skewness(r)
    kurt=kurtosis(r)
    h_var=var_historic(r)
    cF_var=var_gaussian(r,modified=True)
    
    
    
    return pd.DataFrame(
                        {'Annual Ret':a_ret,
                        'Annual Volatility':a_vol,
                        'Sharpe Ratio':s_ratio,
                        'Max Drawdown':Max_drawdown,
                        'Skewness':skw,
                        'Kurtosis':kurt,
                        'Historic VaR (5%)':h_var,
                        'Cornish-Fisher VaR (5%)':cF_var}
                        )






def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    """
    Onlsy run on monthly Data
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 # fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
            
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute the new account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
    risky_wealth = start*(1+risky_r).cumprod()
    
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return cppi

def plot_1(df,title1='Historic Returns',title2="Returns' Distribution"):
  '''
  Requires a Df or series
  '''
  fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
  plt.subplots_adjust(wspace=0.01)
  df.plot(ax=ax1, title=title1)

  if len(df.shape)==1:
    sns.distplot(df,ax=ax2,vertical=True)
    ax2.axhline(y=df.mean(), ls=":", color="Red")
    ax2.axhline(y=df.median(), ls=":", color="blue")
    ax2.annotate(f"Mean: {int(df.mean())}", xy=(.7, .9),xycoords='axes fraction', fontsize=24, color='Red')
    ax2.annotate(f"Median: {int(df.median())}", xy=(.7, .85),xycoords='axes fraction', fontsize=24,color='blue')
    plt.title(title2)  
    plt.xlabel('Frecuency')

  else:
    df.plot.hist(ax=ax2, bins=200,orientation='horizontal',title="Returns' Distribution")





