import pandas as pd
from pandas import datetime
from pandas.io.sql import read_sql_query
import psycopg2
import numpy as np
import sys
import time


from optparse import OptionParser
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os.path
from fpdf import FPDF
from os import listdir
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


start_time = time.time()

parser = OptionParser()

parser.add_option('--s',
                  '--sname',
                  dest='sname',
                  help="timeseries sname for calculating")

parser.add_option('--p',
                  '--portfolio',
                  dest='portfolio',
                  help="portfolio for calculating")

parser.add_option('--fcsv',
                  '--from_csv',
                  dest='from_csv',
                  default=False,
                  help="load data from csv")

parser.add_option('--start',
                  '--start',
                  dest='start',
                  help="start date for calculating",
                  default = "2009-01-01")

parser.add_option('--e',
                  '--end',
                  dest='end',
                  help="end date for calculating",
                  default = pd.Timestamp.now().strftime("%Y-%m-%d"))

parser.add_option('--w',
                  '--window',
                  dest='window',
                  help="sliding_window",
                  default = 15)

parser.add_option('--th',
                  '--threshold',
                  dest='threshold',
                  help="threshold",
                  default = 30)

parser.add_option('--c',
                  '--cores',
                  dest='cores',
                  help="number_of_cores",
                  default = 4 )

options, args = parser.parse_args()





#----------------------------------------------------------------------------
#functions
#---------------------------------------------------------------------------

def connect_to_db(path_to_env='.env'):
    """Establishes connection with the TenViz database.

    Returns `connection` and `cursor` objects.
    """
    try:
        # check if the path exists
        with open(path_to_env, 'r') as f:
            env = dict(line.split('=') for line in f.read().strip().split("\n"))
    except Exception as e:
        raise RuntimeError("Cannot open the file " + path_to_env + ". Reason:\n" + str(e))
    try:
        conn = psycopg2.connect(**{
            "host": env["DB_HOST"],
            "dbname": env["DB_DATABASE"],
            "user": env["DB_USERNAME"],
            "password": env["DB_PASSWORD"],
            "port": env["DB_PORT"]})
        cur = conn.cursor()
        cur.execute("set schema '{schema}';".format(schema=env['DB_SCHEMA']))
    except Exception as e:
        raise RuntimeError("Cannot connect to the database. Reason:\n" + str(e))
    return conn

get_series_query = """
select  ts.id,
        ts.sname,
        date,
        value,
        ts."attributes" ->> 'inverse' as inverse
  from timeseries ts
  join get_series_filling_with_last(ts.id) on true
  join rotation ro on ro.timeseries_id = ts.id
  where date BETWEEN '{start}' and '{end}'
    and ts.sname in ('{ticker}')
    and ro.use_on_frontend = true
    and ro.use_in_analysis = true
"""

get_series_from_portfolio = """
select ts.id,
       ts.sname,
       date,
       value,
       ts."attributes" ->> 'inverse' as inverse
  from timeseries ts
  join get_series_filling_with_last(ts.id)  on true
  join rotation ro on ro.timeseries_id = ts.id
 where ts.id in (select timeseries_id
                   from get_portfolio((select id from portfolio where sname = ({portfolio}))::int))
   and ro.use_on_frontend = true
   and ro.use_in_analysis = true
"""


def get_timeseries(dbc, ticker, start=options.start, end=options.end):
    return read_sql_query(
        get_series_query.format(ticker=ticker, start=start, end=end),
        dbc, parse_dates=['date'])

def get_ts_from_portfolio(dbc, portfolio, start=options.start, end=options.end):
    return read_sql_query(
        get_series_from_portfolio.format(portfolio=portfolio, start=start, end=end),
        dbc)

def pca_date_parser(x):
    return parse_date(x, '%d.%m.%Y')

def macro_date_parser(x):
    return parse_date(x, '%Y-%m-%d')

def parse_date(dt, pattern):
    return datetime.strptime(str(dt), pattern)

threshold = int(options.threshold)

def get_median_filtered(signal, threshold = threshold): #range 10-60 #show difference between signal
#and median in this window (divided on median of this difference) is bigger of some threshold
    """
    signal: is numpy array-like
    returns: signal, numpy array
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

# getting data

if options.from_csv :
    total_frame = pd.read_csv('C:/Users/98250/Documents/R/Russell_check_y.csv', parse_dates=['date'])
elif options.portfolio :
    portfolio_list = str(options.portfolio.split(',')).strip('[').strip(']')
    dbc = connect_to_db(path_to_env='C:/Users/fedel/Desktop/work/.env')
    total_frame =  get_ts_from_portfolio(dbc, portfolio_list).dropna()
    print('## Data load')
    folder = 'w_'+options.window+'_th_'+options.threshold+'_outliers_image_'+pd.Timestamp.now().strftime("%Y-%m-%d")+'_'+options.portfolio
    if total_frame.shape[0] == 0:
        print('### There is no data for this portfolio list')
elif options.sname:
    sname_list = str(options.sname.replace(',',"','")).strip('[').strip(']')
    dbc = connect_to_db(path_to_env='C:/Users/fedel/Desktop/work/.env')
    total_frame =  get_timeseries(dbc, sname_list).dropna()
    print('## Data load')
    folder = 'w_'+options.window+'_th_'+options.threshold+'_outliers_image_'+pd.Timestamp.now().strftime("%Y-%m-%d")+'_'+options.sname
    if total_frame.shape[0] == 0:
        print('### There is no data for this sname list')
else:
    print('### Please add sname/portfolio/fcsv parameter')


id_unique = total_frame['id'].unique()
window_size = int(options.window)# usually, range 10-20, the size of moving window, if the window wide you can catch broad outliers,
#but usually they are tiny, so window should be narrow


all_filtered_data = pd.DataFrame(columns=['id','date', 'value','sname','inverse'])

outliers_fix = pd.DataFrame(columns=['id','date', 'value','sname','inverse'])
outliers_not_fix = pd.DataFrame(columns=['id','date', 'value','sname','inverse'])

def main_loop(ts,outliers_fix, outliers_not_fix,all_filtered_data,folder):
    current_frame = total_frame[total_frame.id == ts]
    snm = total_frame.sname[total_frame.id == ts].unique()
    print("## Searching for outliers in",snm[0], ts)
    value = current_frame.value.astype(float)
    value_s = value.tolist()
    median_filtered_signal = []
    for ii in range(0, value.size, window_size):
        median_filtered_signal += get_median_filtered(np.asanyarray(value_s[ii: ii+window_size])).tolist()
    value = pd.Series(median_filtered_signal)

    date = pd.to_datetime(current_frame['date'])
    value = pd.DataFrame(value).set_index(date.index.values)
    value.columns = ['value']
    sname = current_frame['sname']

    filtered_data = pd.concat([date,value,sname], axis =1,sort = True)
    filtered_data['id'] = ts
    outliers = filtered_data.loc[current_frame['value'] != filtered_data['value']]
    outliers_fix = outliers_fix.append(outliers,sort = True)
    outliers2 = current_frame.loc[current_frame['value'] != filtered_data['value']]
    outliers_not_fix = outliers_not_fix.append(outliers2,sort = True)
    all_filtered_data = all_filtered_data.append(filtered_data,sort = True)
    if not os.path.exists(folder):
           os.mkdir(folder)
    if outliers.shape[0] != 0:
        #Making plots
           fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15, 9), sharex=True)
           ax1.plot(current_frame.date, current_frame.value,color = '#FF5733')
           ax1.plot(outliers_not_fix.date, outliers_not_fix.value, color = '#154360',marker='o',label ='outlier', linestyle = '')
           ax1.legend()

           if 'inverse' in total_frame.columns:
                plt.gca().invert_yaxis()

           fig.text(x=0.45, y=0.95, s=snm[0], fontsize=16, weight='bold', color='#676868')
           fig.text(x=0.05, y=0.5, s="VALUE", fontsize=16, color='#676868', rotation='vertical')
           fig.text(x=0.5, y=0, s="DATE", fontsize=16, color='#676868')
           plt.savefig(folder+'/'+snm[0]+'.png', bbox_inches='tight')
           plt.close(fig)
    return outliers_fix, outliers_not_fix

num_cores = int(options.cores)

results = Parallel(n_jobs=num_cores)(delayed(main_loop)(ts, outliers_fix,outliers_not_fix,all_filtered_data,folder) for ts in id_unique)
outliers_fi=[item[0] for item in results]
outliers_not_fi=[item[1] for item in results]

total_frame = None # delete big variable from memory
imagelist = listdir(folder+'/') # get list of all images

print('## Saving to csv')
total = pd.concat([pd.concat(outliers_not_fi,sort = True), pd.concat(outliers_fi,sort = True)], axis = 1, sort = True)
total.to_csv(folder+'total.csv')


print('## Making pdf')
pdf = FPDF( orientation = 'l', unit = 'mm', format='a4')
for image in imagelist:
    pdf.add_page()
    pdf.image(folder+'/'+image,0, 0, 280, 200)
pdf.output(folder+"images.pdf","f")


print("--- %s min ---" % round(float((time.time() - start_time))/60,2))
