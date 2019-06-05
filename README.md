# Filtering_outliers
python filtering_outliers.py --w 15 --th 30 --portfolio "Credit Spread" --cores 4




METHODOLOGY:
1. calculating median by window
2. calculate the difference between each element and median
3. calculate the median for this difference
4. Check is the difference divided on median of this difference bigger of some threshold

--s,
--sname,          "timeseries sname for calculating"

--p,
--portfolio,      "portfolio for calculating"

--fcsv,
--from_csv,        "load data from csv"

--start,          "start date for calculating"
                  default = "2009-01-01"

--e,
--end,            "end date for calculating"
                  default = pd.Timestamp.now().strftime("%Y-%m-%d")

--w,
--window,         "sliding_window",
                  default = 15

--th,
--threshold,      "threshold",
                  default = 30
--c
--cores,          "number_of_cores",
                  default = 4
