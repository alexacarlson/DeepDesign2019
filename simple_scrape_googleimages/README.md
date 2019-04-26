# Generating a classification dataset by scraping google images 
This code is based upon the project (https://github.com/hardikvasa/google-images-download.git)

These scripts allow you to generate a classification dataset from images generated via google image search. 



## INSTALLATION:
The following dependencies need to be installed: 
python 2.X or python 3.X
Selenium
Chromedriver [Windows](https://sites.google.com/a/chromium.org/chromedriver/downloads)

The path looks like this: "path/to/chromedriver". In windows it will be "C:\path\to\chromedriver.exe"

## USAGE
In a terminal or cmd prompt, navigate to this folder and run"
python generate_architecture_dataset.py --chromedriver_path `pwd`/chromedriver --classes_file_path classes.txt