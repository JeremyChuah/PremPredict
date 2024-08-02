## About this Project

This project is a ML model to predict games for Premier League

Some features include:
* A custom webscraping algorithm with the ability to scrape mutiple seasons worth of data across multiple leagues
* A Machine Learning model trained with a random forest algoirthm able to predict games with 70% accuracy.

## Built With

Built with a Pandas, Scikit-learn, and Beautiful Soup

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Beautiful Soup](https://img.shields.io/badge/Beautiful%20Soup-3776AB?style=for-the-badge&logo=python&logoColor=white)

## Getting Started

To use the project and get started

1. Clone the repo
2. Create a virtual enviornment
   ```
   python3 -m venv env
   ```
   on windows then run
   ```
   .\env\Scripts\activate
   ```
   on mac
   ```
   source env/bin/activate 
   ```
4. The install the required depencies
   ```pip install -r requirements.txt```
5.  In `Webscraper.py` add the years of the seasons you want to scrape data for.
   * Webscrape once for the season you want to predict and once again for the seasons you want to train the data on
     ```
     7. years = [] #add the years you need ot scrape for
     ```
     ```
     9. standings_url = put first season that you are scarping for (i.e. years[0])
     ```
6. After the ```.csv``` files are created, name the one you are training the data on `train.csv` and the one to predict `test.csv`
7. Now run the script in ```predict.py```
8. results will be in `predicted_results.csv`
   ```
   npm start
   ```
## Data Source

This project scrapes data from [FB red](https://fbref.com/en).

<!-- CONTACT -->
## Contact

Your Name - [Linkedin](https://www.linkedin.com/in/jeremy-chuah/) - Jeremychuah15@gmail.com

Project Link: [Repo](https://github.com/JeremyChuah/SubwayUpdate)

<!-- LICENSE -->
## License

Distributed under the MIT License.
