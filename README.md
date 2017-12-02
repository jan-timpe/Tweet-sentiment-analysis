# Tweet sentiment analysis

Machine learning to predict the sentiment of tweets

## Setup and run

Fill out `example_settings.py` with your own credentials and options, then rename the file to `settings.py`

Make sure python3 is installed

Create a virtualenv by running
```$ virtualenv venv -p python3```

Activate the virtualenv by running 
```$ source ./venv/bin/activate```

Install requirements with pip
```$ pip install -r requirements.txt```

Run the sklearn model
```$ python main.py```

Run the spark model
```$ python spark.py```